from random import randint

"""Generating random data"""
def generate_data(length, lower_range=0, upper_range=10000):
    data = []
    for _ in range(0,length):
        data.append(randint(lower_range,upper_range))
    return data

def main():
    print(generate_data(20))

if __name__ == '__main__':
    main()