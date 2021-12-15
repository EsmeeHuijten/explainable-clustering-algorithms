import csv

from numpy import array

from solver_interface import Instance
from util import Point


def get_instance(k: int, parameters: dict[str, bool], num_points: int) -> Instance:
    points = get_points(parameters, num_points)
    print(len(points))
    return Instance(points, k)


def get_points(parameters, num_points: int):
    with open('./data/creditcard/creditcard_data.csv', mode='r') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        data = list(csv_reader)
    clean_data = [cleanup(customer[1], parameters) for customer in enumerate(data) if customer[0] < num_points if
                  cleanup(customer[1], parameters)]
    # print(clean_data)
    points = [Point(array(entry)) for entry in clean_data]
    return points


def cleanup(customer, parameters) -> list:
    cleaned_up = []
    param_names = ['CUST_ID', 'BALANCE', 'BALANCE_FREQUENCY', 'PURCHASES', 'ONEOFF_PURCHASES', 'INSTALLMENTS_PURCHASES',
                   'CASH_ADVANCES', 'PURCHASES_FREQUENCY', 'ONEOFF_PURCHASES_FREQUENCY',
                   'PURCHASES_INSTALLMENTS_FREQUENCY', 'CASH_ADVANCE_FREQUENCY', 'CASH_ADVANCE_TRX', 'PURCHASES_TRX',
                   'CREDIT_LIMIT', 'PAYMENTS', 'MINIMUM_PAYMENTS', 'PRC_FULL_PAYMENT', 'TENURE']
    for name in param_names:
        if parameters[name]:
            data = float(customer[name])
            dataraw = customer[name]
            # print(type(dataraw))
            # if customer[name] != "0":
            #     data = float(customer[name])
            # else:
            #     print("empty cell detected")
            #     return False
            cleaned_up.append(data)

    return cleaned_up
