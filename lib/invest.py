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
from typing import Optional, Union, List, Dict, Iterable
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
        self._free_usd = 0.0
        self._expected_portfolio_cost = 5000.0

    @staticmethod
    def _fix_tickers(tickers: str):
        return tickers.replace('.', '-')

    @staticmethod
    def _get_historical_price(tickers: str, period: str, interval: str, price_type: str = 'Close') -> pd.DataFrame:
        tickers = Investments._fix_tickers(tickers).split()

        prices = []
        for ticker in tickers:
            while True:
                try:
                    info = yf.download(ticker, period=period, progress=False, prepost=True, interval=interval, threads=False)
                    info = info[price_type]
                    break
                except:
                    logging.warning(f'Failed to fetch current price for {ticker}, retrying in 1 second')
                    time.sleep(1)
            prices.append(info)
        result = pd.concat(prices, axis=1, join='outer')
        result.columns = tickers
        result = result.fillna(method='ffill').fillna(method='bfill').fillna(value=0.0)
        return result

    @staticmethod
    def _get_today_price(tickers: str, price_type: str = 'Close') -> Union[float, List[float]]:
        tickers = Investments._fix_tickers(tickers)
        info = Investments._get_historical_price(tickers, period='5d', interval='1h', price_type=price_type)
        today_row = info.iloc[-1, :]
        prices = list(today_row)
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

    def _get_operations(self):
        client = tinvest.SyncClient(
            token=self._tinkoff_token
        )
        return client.get_operations(from_=datetime.fromtimestamp(0), to=datetime.utcnow()).payload.operations

    def _get_total_buy_price(self):
        operations = self._get_operations()
        total_buy_price = 0.0
        for operation in operations:
            if operation.status != tinvest.OperationStatus.done:
                continue

            if operation.instrument_type == tinvest.InstrumentType.currency and operation.figi == 'BBG0013HGFT4':  # usd
                if operation.operation_type in [tinvest.OperationType.buy, tinvest.OperationTypeWithCommission.buy]:
                    total_buy_price += operation.quantity
                if operation.operation_type in [tinvest.OperationType.sell, tinvest.OperationTypeWithCommission.sell]:
                    total_buy_price -= operation.quantity
                continue

            if operation.currency == tinvest.Currency.usd and operation.operation_type == tinvest.OperationTypeWithCommission.pay_in:
                total_buy_price += float(operation.payment)
                continue
        return total_buy_price

    def _get_total_current_price(self):
        return sum(stock.current_price * stock.my_lots for stock in self._stocks) + self._free_usd

    def update_stock_info(self):
        all_positions = self._get_positions()
        stock_positions = list(filter(lambda pos: pos.instrument_type == tinvest.InstrumentType.stock and
                                                  pos.average_position_price.currency == tinvest.Currency.usd,
                                      all_positions))

        self._free_usd = 0.0
        for pos in all_positions:
            if pos.instrument_type == tinvest.InstrumentType.currency:
                self._free_usd = float(pos.balance)

        today_prices = [float((pos.average_position_price.value * pos.lots + pos.expected_yield.value) / pos.lots) for pos in stock_positions]

        current_portfolio_cost = sum(pos.lots * price for pos, price in zip(stock_positions, today_prices))
        while self._expected_portfolio_cost < current_portfolio_cost:
            self._expected_portfolio_cost *= 1.5
        self._expected_portfolio_cost = ceil(self._expected_portfolio_cost / 5000.0) * 5000.0

        stocks = self._sample_sp100(self._expected_portfolio_cost)
        for pos, price in tqdm(zip(stock_positions, today_prices)):
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

        wks.clear(fields='*')

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

        total_buy_price = self._get_total_buy_price()
        total_current_price = self._get_total_current_price()
        total_yield = total_current_price - total_buy_price

        info_cells = [
            pygsheets.Cell(pos='H2', val=f'Portfolio buy price:'),
            pygsheets.Cell(pos='I2', val=total_buy_price),
            pygsheets.Cell(pos='H3', val=f'Portfolio current price:'),
            pygsheets.Cell(pos='I3', val=total_current_price),
            pygsheets.Cell(pos='H4', val=f'Portfolio yield:'),
            pygsheets.Cell(pos='I4', val=total_yield),
            pygsheets.Cell(pos='H5', val=f'Target price:'),
            pygsheets.Cell(pos='I5', val=self._expected_portfolio_cost)
        ]

        for i, cell in enumerate(info_cells):
            if i % 2 == 0:
                cell.set_text_format('bold', True)
            else:
                cell.set_number_format(pygsheets.FormatType.NUMBER, pattern='0.0')

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
        total_buy_price = self._get_total_buy_price()
        total_current_price = self._get_total_current_price()
        today = datetime.today().strftime('%Y-%m-%dT%H:%M:%S')

        sp100_current_price = self._get_today_price('^OEX', 'Close')
        if sp100_current_price < 1.0:
            return

        with Path(self._history_path).open('a') as fp:
            print(f'{today}\t{total_buy_price}\t{total_current_price}\t{sp100_current_price}', file=fp)

    @staticmethod
    def visualize_ema(tickers: List[str]):
        tickers = list(map(Investments._fix_tickers, tickers))
        unique_tickers = list(set(tickers))
        prices = Investments._get_historical_price(' '.join(unique_tickers), period='3mo', interval='1h')
        total_price = sum(prices[ticker] for ticker in tickers)

        dates = [date.strftime('%Y-%m-%d %H:%M:%S') for date in prices.index]
        ema_5d = total_price.ewm(span=16*5).mean()
        ema_1mo = total_price.ewm(span=16*21).mean()

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=dates, y=total_price.to_numpy(), mode='lines', name='price'))
        fig.add_trace(go.Scatter(x=dates, y=ema_5d.to_numpy(), mode='lines', name='ema_5d'))
        fig.add_trace(go.Scatter(x=dates, y=ema_1mo.to_numpy(), mode='lines', name='ema_1mo'))
        img = fig.to_image(format='png')
        return img

    def visualize_history(self):
        df = pd.read_csv(self._history_path, sep='\t')

        total_buy_price = df['total_buy_price'].to_numpy()
        total_current_price = df['total_current_price'].to_numpy()
        portfolio = np.ones_like(total_buy_price)
        multiplier = np.ones_like(total_buy_price)
        sp100 = df['sp100_price'] / df.loc[0, 'sp100_price']

        coef, m = total_buy_price[0] / total_current_price[0], 1.0
        for i in range(1, len(portfolio)):
            if abs(total_buy_price[i] - total_buy_price[i - 1]) > 1.0:
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
