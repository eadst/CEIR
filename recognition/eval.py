import glob
import os

main_path = os.path.abspath(os.path.join(os.getcwd(), "../.."))
gt_path = os.path.join(main_path, 'result/step3/label/')
filenames1 = [os.path.splitext(f)[0] for f in glob.glob("data_test_gt/*.txt")]
txt_gt_files = [s + ".txt" for s in filenames1]
# txt_gt_list = [path.split('\\')[-1] for path in txt_gt_files]

filenames2 = [os.path.splitext(f)[0] for f in glob.glob("data_test_result/*.txt")]
txt_result_files = [s + ".txt" for s in filenames2]
# txt_result_list = [path.split('\\')[-1] for path in txt_result_files]
word_count = 0
match_count = 0
for result_path in txt_result_files:
    gt_path = 'data_test_gt\\' + result_path.split('\\')[-1]
    if gt_path in txt_gt_files:
        result = open(result_path)
        gt = open(gt_path)
        for result_line in result.readlines():
            for gt_line in gt.readlines():
                result_list = result_line.split()
                gt_list = gt_line.split()
                word_count += len(result_list)
                for word in result_list:
                    if word.upper() in gt_list:
                        match_count += 1
    else:
        print('cannot find: ', result_path)

print('word_count: ', word_count)
print('match_count: ', match_count)
print('precision: ', match_count/float(word_count))

