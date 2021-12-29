import tinvest
import pandas as pd
import requests
import yfinance as yf
import numpy as np
import copy
import time
import traceback
import logging

from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Union, List, Dict
from threading import Thread, RLock
from math import ceil
from tqdm import tqdm

import pygsheets
import telegram
from telegram.ext import Updater

import plotly.graph_objects as go


from .config import TINKOFF_API_TOKEN, GOOGLE_SA_CONFIG, BOT_TOKEN, PERSONAL_CHAT_ID


@dataclass
class Stock:
    name: str
    ticker: str
    current_price: float
    sp100_weight: float
    sp100_lots: int
    my_lots: int
    avg_buy_price: Optional[float]

    @property
    def total_price(self) -> float:
        return self.current_price * self.my_lots

    @property
    def profit(self) -> float:
        return self.current_price / self.avg_buy_price - 1 if self.avg_buy_price else 0.0

    @property
    def expected_yield(self) -> float:
        return (self.current_price - self.avg_buy_price) * self.my_lots if self.avg_buy_price else 0.0

    def __copy__(self) -> 'Stock':
        return Stock(self.name, self.ticker, self.current_price, self.sp100_weight, self.sp100_lots, self.my_lots, self.avg_buy_price)


class Investments:
    def __init__(self, tinkoff_token: str, google_sa_config: str, history_path: str):
        self._tinkoff_token = tinkoff_token
        self._google_sa_config = google_sa_config
        self._history_path = history_path
        self._stocks = []
        self._expected_portfolio_cost = 5000.0

    @staticmethod
    def _get_today_price(tickers: str, price_type: str = 'Close') -> Union[float, List[float]]:
        tickers = tickers.replace('.', '-').split()
        prices = []
        for ticker in tickers:
            if ticker in ['SPBE']:
                continue
            while True:
                try:
                    info = yf.download(ticker, period='5d', progress=False, prepost=True, interval='1h', threads=False)
                    break
                except:
                    time.sleep(1)
            today_row = info.fillna(method='ffill').iloc[-1, :]
            open_price = today_row[price_type]
            prices.append(float(open_price))
        return prices[0] if len(prices) == 1 else prices

    @staticmethod
    def _sample_sp100(expected_cost: float) -> Dict[str, Stock]:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_3) AppleWebKit/537.75.14 (KHTML, like Gecko) Version/7.0.3 Safari/7046A194A',
            'X-Requested-With': 'XMLHttpRequest'
        }
        resp = requests.get('https://www.slickcharts.com/sp500', headers=headers)

        sp500 = pd.read_html(resp.text)[0]
        sp100 = sp500[:100]

        # price * lots / budget = weight
        samples = pd.DataFrame(columns=['Company', 'Ticker', 'Price', 'Weight', 'Lots'])
        samples['Company'] = sp100['Company']
        samples['Ticker'] = sp100['Symbol']
        samples['Price'] = Investments._get_today_price(' '.join(samples['Ticker'].to_numpy()))
        samples['Weight'] = sp100['Weight'] / sp100['Weight'].sum()
        samples['Lots'] = (samples['Weight'] * expected_cost / samples['Price']).round().astype(int)

        stocks = {}
        for _, row in samples.iterrows():
            stocks[row['Ticker']] = Stock(
                name=row['Company'],
                ticker=row['Ticker'],
                current_price=row['Price'],
                sp100_weight=row['Weight'],
                sp100_lots=row['Lots'],
                my_lots=0,
                avg_buy_price=None
            )
        return stocks

    def _get_positions(self):
        client = tinvest.SyncClient(
            token=self._tinkoff_token
        )
        return client.get_portfolio().payload.positions

    def update_stock_info(self):
        positions = self._get_positions()
        positions = list(filter(lambda pos: pos.instrument_type == tinvest.InstrumentType.stock and
                                            pos.average_position_price.currency == tinvest.Currency.usd,
                                positions))

        # tickers = ' '.join(pos.ticker for pos in positions)
        # today_prices = self._get_today_price(tickers, price_type='Close')
        today_prices = [float((pos.average_position_price.value * pos.lots + pos.expected_yield.value) / pos.lots) for pos in positions]

        current_portfolio_cost = sum(pos.lots * price for pos, price in zip(positions, today_prices))
        while self._expected_portfolio_cost < current_portfolio_cost:
            self._expected_portfolio_cost *= 1.5
        self._expected_portfolio_cost = ceil(self._expected_portfolio_cost / 5000.0) * 5000.0

        stocks = self._sample_sp100(self._expected_portfolio_cost)
        for pos, price in tqdm(zip(positions, today_prices)):
            if pos.ticker in stocks:
                stocks[pos.ticker].my_lots = pos.lots
                stocks[pos.ticker].avg_buy_price = float(pos.average_position_price.value)
            else:
                stocks[pos.ticker] = Stock(
                    name=pos.name,
                    ticker=pos.ticker,
                    current_price=price,
                    sp100_weight=0.0,
                    sp100_lots=0,
                    my_lots=pos.lots,
                    avg_buy_price=float(pos.average_position_price.value)
                )

        ordered_stocks = list(filter(lambda stock: stock.my_lots > 0 or stock.sp100_lots > 0, stocks.values()))
        ordered_stocks.sort(key=lambda stock: (-stock.sp100_weight, stock.ticker))
        self._stocks = ordered_stocks

    def update_spreadsheet(self):
        gc = pygsheets.authorize(client_secret=None, service_account_file=self._google_sa_config)

        sh: pygsheets.Spreadsheet = gc.open('invest')
        wks: pygsheets.Worksheet = sh.sheet1

        wks.clear()

        header: pygsheets.DataRange = wks.range('A1:E1', returnas='range')
        for cell in header.cells[0]:
            cell.set_text_format('bold', True)
        header.update_values([['Company', 'Ticker', 'Price', 'Lots', 'Profit']])

        data = []
        for i, stock in enumerate(self._stocks):
            row = [
                pygsheets.Cell(pos=f'A{i+2}', val=stock.name),
                pygsheets.Cell(pos=f'B{i+2}', val=stock.ticker),
                pygsheets.Cell(pos=f'C{i+2}', val=stock.current_price),
                pygsheets.Cell(pos=f'D{i+2}', val=f'{stock.my_lots}/{stock.sp100_lots}'),
                pygsheets.Cell(pos=f'E{i+2}', val=stock.profit)
            ]

            row[2].set_number_format(pygsheets.FormatType.NUMBER, pattern='0.00')

            lw = stock.my_lots / max(stock.my_lots, stock.sp100_lots)
            row[3].color = (1.0, lw * 2, 0, 0.8) if lw < 0.5 else (2 - lw * 2, 1.0, 0, 0.8)
            row[3].set_horizontal_alignment(pygsheets.HorizontalAlignment.RIGHT)

            pw = min(1, 0.5 + 0.5 * stock.profit)
            row[4].color = (1.0, pw * 2, pw * 2, 0.8) if pw < 0.5 else (2 - pw * 2, 1.0, 2 - pw * 2, 0.8)
            row[4].set_number_format(pygsheets.FormatType.PERCENT, pattern='0.00%')

            data.extend(row)
        wks.update_cells(data)

        total_weight = sum(stock.total_price for stock in self._stocks)
        total_yield = sum(stock.expected_yield for stock in self._stocks)

        info_cells = [
            pygsheets.Cell(pos='H2', val=f'Portfolio cost:'),
            pygsheets.Cell(pos='I2', val=total_weight),
            pygsheets.Cell(pos='H3', val=f'Portfolio yield:'),
            pygsheets.Cell(pos='I3', val=total_yield),
            pygsheets.Cell(pos='H4', val=f'Target cost:'),
            pygsheets.Cell(pos='I4', val=self._expected_portfolio_cost)
        ]

        info_cells[0].set_text_format('bold', True)
        info_cells[1].set_number_format(pygsheets.FormatType.NUMBER, pattern='0.0')
        info_cells[2].set_text_format('bold', True)
        info_cells[3].set_number_format(pygsheets.FormatType.NUMBER, pattern='0.0')
        info_cells[4].set_text_format('bold', True)
        info_cells[5].set_number_format(pygsheets.FormatType.NUMBER, pattern='0.0')

        wks.update_cells(info_cells)

        return sh.url

    def suggest_shares(self, budget: float) -> List[str]:
        shares = []
        stocks = {stock.ticker: copy.deepcopy(stock) for stock in self._stocks}

        def suggest_one_share() -> Optional[str]:
            weights = np.zeros(len(stocks))
            tickers = list(stocks.keys())
            for i, ticker in enumerate(tickers):
                stock = stocks[ticker]
                if stock.current_price > budget or stock.my_lots >= stock.sp100_lots:
                    continue
                weights[i] = (stock.sp100_lots - stock.my_lots) * stock.current_price

            if weights.sum() == 0.0:
                return None

            weights /= weights.sum()
            return np.random.choice(tickers, p=weights)

        while True:
            share_to_buy = suggest_one_share()
            if share_to_buy is None:
                break
            shares.append(share_to_buy)
            budget -= stocks[share_to_buy].current_price
            stocks[share_to_buy].my_lots += 1

        return shares

    def update_history(self):
        total_buy_price = sum(stock.avg_buy_price * stock.my_lots for stock in self._stocks if stock.my_lots > 0)
        total_current_price = sum(stock.current_price * stock.my_lots for stock in self._stocks)
        today = datetime.today().strftime('%Y-%m-%dT%H:%M:%S')

        sp100_current_price = self._get_today_price('^OEX', 'Close')
        if sp100_current_price < 1.0:
            return

        with Path(self._history_path).open('a') as fp:
            print(f'{today}\t{total_buy_price}\t{total_current_price}\t{sp100_current_price}', file=fp)

    def visualize_history(self):
        df = pd.read_csv(self._history_path, sep='\t')

        total_buy_price = df['total_buy_price'].to_numpy()
        total_current_price = df['total_current_price'].to_numpy()
        portfolio = np.ones_like(total_buy_price)
        multiplier = np.ones_like(total_buy_price)
        sp100 = df['sp100_price'] / df.loc[0, 'sp100_price']

        coef, m = total_buy_price[0] / total_current_price[0], 1.0
        for i in range(1, len(portfolio)):
            if total_buy_price[i] - total_buy_price[i - 1] > 100.0:
                m *= (total_current_price[i - 1] / total_buy_price[i - 1]) / (total_current_price[i] / total_buy_price[i])
            portfolio[i] = total_current_price[i] / total_buy_price[i] * coef * m
            multiplier[i] = m

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df['date'], y=multiplier, mode='lines', name='multiplier', line=dict(color='green')))
        fig.add_trace(go.Scatter(x=df['date'], y=sp100, mode='lines', name='sp100', line=dict(color='red')))
        fig.add_trace(go.Scatter(x=df['date'], y=portfolio, mode='lines', name='portfolio', line=dict(color='blue')))
        img = fig.to_image(format='png')
        return img


def run_invest_updater(debug=False):
    def update():
        try:
            logging.info('Updating stock info and history')
            investments.update_stock_info()
            investments.update_history()
        except Exception as e:
            status_text = f'<b>Error</b>: Failed to update stock info\n' \
                          f'<b>Exception</b>: {e}\n' \
                          f'<b>Traceback</b>: {traceback.format_exc()}'
            Updater(BOT_TOKEN).bot.send_message(PERSONAL_CHAT_ID, status_text, parse_mode=telegram.ParseMode.HTML)

    def track():
        while True:
            if debug:
                update()
            else:
                update_thread = Thread(target=update, daemon=False)
                update_thread.start()
            time.sleep(1 * 60 * 60)

    if debug:
        track()
    else:
        subthread = Thread(name='invest_updater', target=track, daemon=False)
        subthread.start()


investments = Investments(TINKOFF_API_TOKEN, GOOGLE_SA_CONFIG, 'data/history.tsv')
