import numpy as np

def mat2file(image, kappa, file):
    for x in image:
        file.write(' '.join(map(str, x)))
        file.write(' ')
    file.write(';')
    for x in kappa:
        file.write(' '.join(map(str, x)))
        file.write(' ')
    file.write('\n')


def file2mat(directory):
    result = [[], []]
    with open(directory, 'r') as file:
        for line in file:
            line = line.split(';')
            for i, mat in enumerate(line):
                mat = np.fromiter(map(float, mat.split()), dtype='float')
                size = np.sqrt(len(mat))
                assert size.is_integer()
                size = int(size)
                result[i].append(mat.reshape((size, size)).tolist())
    return result


# for i in range(3):
#     x = np.random.randint(1, 6, (3, 3)).tolist()
#     y = np.random.uniform(1, 6, (3, 3)).tolist()
#     mat2file(x, y, 'test.txt')
#
# image, kappa = file2mat('test.txt')
# print(kappa)
