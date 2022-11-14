def load_txt_to_dic(file_txt):
    result_id_number = dict()
    result_id_label = dict()
    with open(file_txt, 'r', encoding='utf-8') as f:
        context = f.read().strip().split('\n')
    for x in context:
        line = x.split(" ", 2)
        result_id_number[int(line[0])] = int(line[1])
        result_id_label[int(line[0])] = line[2]

    return result_id_number, result_id_label

if __name__=="__main__":
    print(load_txt_to_dic("./data/map_level_0.txt"))
    print(load_txt_to_dic("./data/map_level_1.txt"))
    print(load_txt_to_dic("./data/map_level_2.txt"))
