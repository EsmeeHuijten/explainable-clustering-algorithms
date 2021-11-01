import csv

from numpy import array

from solver_interface import Instance
from util import Point


def get_instance(k: int, parameters: dict[str, bool]) -> Instance:
    points = get_points(parameters)
    return Instance(points, k)


def get_points(parameters):
    with open('./data/mall/Mall_Customers.csv', mode='r') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        data = list(csv_reader)
    clean_data = [cleanup(customer, parameters) for customer in data]
    points = [Point(array(entry)) for entry in clean_data]
    return points


def cleanup(customer, parameters) -> list:
    cleaned_up = []
    if parameters['age']:
        cleaned_up.append(int(customer['Age']))
    if parameters['income']:
        cleaned_up.append(int(customer['Annual Income (k$)']))
    if parameters['spending_score']:
        cleaned_up.append(int(customer['Spending Score (1-100)']))
    return cleaned_up
