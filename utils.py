import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def buy_short_stock(index, state, action, stock_dimension, buy_cost_list):  # expect price up

    buy_num_shares = min(
        action, abs(state[index + stock_dimension + 2 * 3])
    )
    buy_amount = (
            state[index + 2 * 3]
            * buy_num_shares
    )
    # update balance
    state[3] += buy_amount  # limit
    state[4] -= buy_amount * buy_cost_list[index]  # credit balance
    state[5] -= buy_amount * buy_cost_list[index]  # equity
    state[index + stock_dimension + 2 * 3] += buy_num_shares  # holding shares

    return state

def sell_long_stock(index, state, action, stock_dimension, sell_cost_list):

    sell_num_shares = min(
        abs(action), state[index + stock_dimension + 2*3]
    )
    sell_amount = (
        state[index + 2*3]
        * sell_num_shares
        * (1 - sell_cost_list[index])
    )

    # update balance
    state[0] += sell_amount # cash
    state[2] -= state[index + 2*3] * sell_num_shares * sell_cost_list[index]
    state[index + stock_dimension + 2*3] -= sell_num_shares # holding shares

    return state


def update_ratio(date, state, change_ratio, stock_dimension, buy_cost_list, sell_cost_list):
    market = np.array(state[2 * 3: (stock_dimension + 2 * 3)]) * np.array(
        state[(stock_dimension + 2 * 3): (stock_dimension * 2 + 2 * 3)])
    argsort_market = np.argsort(market)

    # Update ratio
    sum_equity = state[2] + state[5]
    current_long_short_ratio = state[2] / sum_equity
    target_long_short_ratio = current_long_short_ratio * (1 + change_ratio)

    # ratio can only be between 0.1 to 0.9
    target_long_short_ratio = min(0.9, target_long_short_ratio)
    target_long_short_ratio = max(0.1, target_long_short_ratio)

    if target_long_short_ratio == current_long_short_ratio:
        pass

    if target_long_short_ratio < 0 or target_long_short_ratio > 1:
        pass

    # calculate target long equity
    target_long_equity = target_long_short_ratio * sum_equity
    long_diff = state[2] - target_long_equity

    # Move long to short
    if long_diff >= 0:
        # threshold = actual cash
        threshold = state[0] - state[1]
        # if actual cash < diff
        if threshold < long_diff:
            argsort_long = argsort_market[-np.where(market > 0)[0].shape[0]:]
            # sell long stock
            for i in argsort_long:
                state = sell_long_stock(i, state, state[stock_dimension + 2 * 3 + i], stock_dimension, sell_cost_list)
                # update threshold after sell
                threshold = state[0] - state[1]
                # if consider transaction fee, update diff after sell
                long_diff = state[2] - (state[2] + state[5]) * target_long_short_ratio
                # until actual cash >= diff
                if threshold >= long_diff:
                    break
        # if actual cash >= diff or have sold all stocks
        diff = min(threshold, long_diff)
        state[0] -= 2 * diff
        state[1] -= diff
        state[2] -= diff
        state[3] += 2 * diff
        state[4] += 3 * diff
        state[5] += diff
    # Move short to long
    else:
        short_diff = abs(long_diff)
        # threshold = available limit/2
        threshold = state[3] / 2
        # if available limit/2 < diff
        if threshold < short_diff:
            # buy short stock
            argsort_short = argsort_market[:np.where(market < 0)[0].shape[0]][::-1]
            for i in argsort_short:
                state = buy_short_stock(i, state, abs(state[2 * 3 + stock_dimension + i]), stock_dimension, buy_cost_list)
                # update threshold after buy
                threshold = state[3] / 2
                # if consider transaction fee, update diff after buy
                long_diff = state[2] - (state[2] + state[5]) * target_long_short_ratio
                short_diff = abs(long_diff)
                if threshold >= short_diff:
                    break
        # if available limit/2 >= diff or have brought all stocks
        diff = min(threshold, short_diff)
        state[0] += 2 * diff
        state[1] += diff
        state[2] += diff
        state[3] -= 2 * diff
        state[4] -= 3 * diff
        state[5] -= diff

    return state, target_long_short_ratio

def data_split_include(df, start, end, target_date_col="date"):
    data = df[(df[target_date_col] >= start) & (df[target_date_col] <= end)]
    data = data.sort_values([target_date_col, "tic"], ignore_index=True)
    data.index = data[target_date_col].factorize()[0]
    return data


def plot_equity(read_path, save_path):
    titlename = "Equity changes"

    result = pd.read_csv(read_path)
    result['date'] = pd.to_datetime(result['date'])
    result.set_index('date', inplace=True)

    plt.figure(figsize=(10, 6))
    plt.plot(result.index, result['wo_llm'], label='Margin Trader (w/o LLM)')
    plt.plot(result.index, result['w_llm'], label='Bi-annual')
    plt.plot(result.index, result['dji'], label='DJI', linestyle='--')

    plt.legend(fontsize=16, framealpha=0.5)
    plt.xlabel('Date', fontsize=16)
    plt.ylabel('Equity', fontsize=16)
    plt.title(titlename, fontsize=16)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.gca().yaxis.get_offset_text().set_fontsize(16)

    # Display the grid
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(save_path, format='pdf')
