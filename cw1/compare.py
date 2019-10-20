import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", type=str)
    parser.add_argument("--o", type=str)
    args = parser.parse_args()

    with open(args.m, 'r', encoding='utf-8') as f1:
        r1 = f1.readlines()

    with open(args.o, 'r', encoding='utf-8') as f2:
        r2 = f2.readlines()

    assert len(r1) == len(r2)
    for i in range(len(r1)):
        if not r1[i] == r2[i]:
            print(r1[i], '|', i, '|', r2[i])
        # if i > 100:
            # break
    if r1 == r2:
        print('So good!')
    else:
        print('Wrong!')
