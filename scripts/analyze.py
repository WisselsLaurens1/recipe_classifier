import json


def class_ratio(Y):
    ratios = {}
    result = []
    tot = 0
    for y in Y:
        ratios.setdefault(y, 0)
        ratios[y] += 1
        tot += 1
    for key in ratios:
        ratios[key] = ratios[key]/tot
        result.append((key, tot*ratios[key], ratios[key]))

    result.sort(key=lambda x: x[1], reverse=True)
    return result


def main():
    PATH = './data/train.json'

    with open(PATH, 'r') as f:
        raw = f.read()
        data = json.loads(raw)

    cuisines = [sample['cuisine'] for sample in data]

    ratios = class_ratio(cuisines)

    for r in ratios:
        print(f'*{r[0]} {r[1]}')


if __name__ == '__main__':
    main()
