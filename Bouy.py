"""from bs4 import BeautifulSoup
import urllib2"""
import os
import requests

down_link = "http://www.ndbc.noaa.gov/data/historical/stdmet/"
source_file = "/home/skanda/Bouy_Data/cluster_1/2017/data_meta/Cluster_1.txt"

cur_dir = home_dir = '/home/skanda/Bouy_Data'  # change accordingly
hdr = {
    'Cookie': 'prov=188c8cb0-02ba-2a19-b655-76a21dd4adc2; __qca=P0-1668401742-1483638160855; _ga=GA1.2.2105094679.1483638161; hero=views=!',
    'Accept-Encoding': 'none',
    'User-Agent': 'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:52.0) Gecko/20100101 Firefox/52.0',
    'Accept': 'text/html,application'
}

def download(url, name):
    file_path = os.path.join(cur_dir, name)
    if os.path.exists(file_path):
        print "hi"
        return 0
    _req = requests.get(url, stream=True, headers = hdr)
    print _req.status_code
    print url.split('2017')[1]
    if(_req.status_code == 200):
        with open(file_path, 'wb') as hand:
            for block in _req.iter_content(chunk_size=1024):
                if block:
                    hand.write(block)
                    #print block
                    #print 'suc3'
            hand.close()
    print '\n -- FILE ' + name + ' DOWNLOADED -- \n', 


def make_dir(cur_dir):
    if not os.path.exists(cur_dir):
        os.mkdir(cur_dir)

def process(link):
    link_end = 'h' + year + '.txt.gz'
    name = link.split('=')[1]
    name = name.lower()

    _link = down_link + name + link_end
    print _link

    download(_link, name)


print "in main"
make_dir(cur_dir)
cluster = raw_input('Enter Cluster Name : ')
cluster = "cluster_"+cluster
cur_dir = os.path.join(cur_dir, cluster)
make_dir(cur_dir)

year = raw_input('Enter year : ')
cur_dir = os.path.join(cur_dir, year)
make_dir(cur_dir)

with open(source_file, "r") as hand:
    lines  = hand.readlines()
    for line in lines:
        process(line.strip())

print "DONE"

# _REQUEST_LINK_STRUCTURE = http://www.ndbc.noaa.gov/data/historical/stdmet/nblp1h2017.txt.gz
