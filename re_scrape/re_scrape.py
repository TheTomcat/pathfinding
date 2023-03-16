from dataclasses import dataclass
from collections import namedtuple
import csv
import re
import os

from bs4 import BeautifulSoup
import requests

input_dir = "re_scrape\\results"
output_file = "re_scrape/output.csv"


fields= ["id","address","url","size","beds","baths","cars","price","sold_on","property_type"]
House = namedtuple("House",fields)
with open(output_file, 'w') as w:   
    count = 0
    for input_file in os.scandir(input_dir):
        with open(os.path.join(input_dir,input_file.name), 'rb') as r:
            print(f"reading {input_file}")
            soup = BeautifulSoup(r.read(), "html.parser")
            properties = soup.find_all(class_="residential-card__content-wrapper")
            writer = csv.writer(w, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL)
            writer.writerow(fields)
            for i, prop in enumerate(properties):
                address_url = prop.find("a", class_="residential-card__details-link")
                address = address_url.find("span")
                beds = prop.find("span", class_="general-features__beds")
                baths = prop.find("span", class_="general-features__baths")
                cars = prop.find("span", class_="general-features__cars")
                size = prop.find("span", class_="property-size__land") or prop.find("span", class_="property-size__building")
                price = prop.find("span", class_="property-price")
                prop_type = prop.find("span", class_="residential-card__property-type")
                try:
                    sold_on = re.search(r"Sold on ([\w\d\s]*)", prop.text).group(1)
                except:
                    sold_on = ""
                try:
                    house = House(
                        id=count+1,
                        url=address_url["href"],
                        address=address.text if address else "",
                        beds=beds.text if beds else "",
                        baths=baths.text if baths else "",
                        cars=cars.text if cars else "",
                        size=size.text if size else "",
                        price=price.text if price else "",
                        sold_on=sold_on if sold_on else "",
                        property_type=prop_type.text if prop_type else ""
                    )
                    writer.writerow(house)
                except AttributeError as e:
                    print(e)
                    # i = input("raise?")
                    # if i=="y":
                    #     raise e
                count+= 1
