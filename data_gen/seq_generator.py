import random

if __name__ == "__main__":
    y = []
    for deg in range(10, 26):
        with open(f'inputs/input_{deg}.txt', 'w') as out:
            for i in range(1 << deg):
                out.write(str(random.randint(-100, 100)) + ' 0\n')
