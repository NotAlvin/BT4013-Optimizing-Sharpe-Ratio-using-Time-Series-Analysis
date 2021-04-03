def mean_difference(filename):
    my_file = open(filename, "r")
    content_list = my_file. readlines()
    content_list = list(map(lambda x: float(x[:-2]), content_list))
    mean = sum(content_list)/len(content_list)
    return mean
print(mean_difference('storage.txt'))



