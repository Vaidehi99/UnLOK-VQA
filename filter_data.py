import json
import os

x = json.load(open("/nas-ssd2/vaidehi/nlp13/belief-localization/third_party/data/zsre_mend_eval.json", "r"))
img_path_easy = "/nas-ssd2/vaidehi/nlp13/data/parap_images/okvqa/{}.jpg"
img_path_medium = "/nas-ssd2/vaidehi/nlp13/data/parap_images_medium/okvqa/{}.jpg"
img_path_hard = "/nas-ssd2/vaidehi/nlp13/data/paraphrase_images_hard/okvqa_yolo_dino_bert/processed_images/{}.jpg"

#for m in x:
#img_path_easy_new = img_path_easy.format(m['sample_id'])
#img_path_easy_medium_new = img_path_medium.format(m['sample_id'])
#img_path_easy_hard_new = img_path_hard.format(m['sample_id'])

img_path_easy_neigh = "/nas-ssd2/vaidehi/nlp13/data/neigh_images_easy/okvqa/{}_{}.jpg"
img_path_medium_neigh = "/nas-ssd2/vaidehi/nlp13/data/neigh_images_medium/okvqa/{}_{}.jpg"
img_path_hard_neigh = "/nas-ssd2/vaidehi/nlp13/data/neighborhood_images_hard/okvqa/{}_{}.jpg"

x = x[:700]
print(len(x))
eval_samples = []
for m in x:
	img_path_easy_new = img_path_easy.format(m['id'])
	img_path_medium_new = img_path_medium.format(m['id'])
	img_path_hard_new = img_path_hard.format(m['id'])
	img_path_easy_neigh_new = img_path_easy_neigh.format(m['id'], 0)
	img_path_medium_neigh_new = img_path_medium_neigh.format(m['id'], 0)
	img_path_hard_neigh_new = img_path_hard_neigh.format(m['id'], 0)
	img_path_easy_neigh_new_2 = img_path_easy_neigh.format(m['id'], 1)
	img_path_medium_neigh_new_2 = img_path_medium_neigh.format(m['id'], 1)
	img_path_hard_neigh_new_2 = img_path_hard_neigh.format(m['id'], 1)
	img_path_easy_neigh_new_3 = img_path_easy_neigh.format(m['id'], 2)
	img_path_medium_neigh_new_3 = img_path_medium_neigh.format(m['id'], 2)
	img_path_hard_neigh_new_3 = img_path_hard_neigh.format(m['id'], 2)

	if not (os.path.exists(img_path_easy_new) and os.path.exists(img_path_medium_new) and os.path.exists(img_path_hard_new) 
		and os.path.exists(img_path_easy_neigh_new) and os.path.exists(img_path_medium_neigh_new) and os.path.exists(img_path_hard_neigh_new)
		and os.path.exists(img_path_easy_neigh_new_2) and os.path.exists(img_path_medium_neigh_new_2) and os.path.exists(img_path_hard_neigh_new_2)
		and os.path.exists(img_path_easy_neigh_new_3) and os.path.exists(img_path_medium_neigh_new_3) and os.path.exists(img_path_hard_neigh_new_3)):
		continue
	else:
		eval_samples.append(m)

print(len(eval_samples))

json.dump(eval_samples, open("/nas-ssd2/vaidehi/nlp13/belief-localization/third_party/data/human_eval_samples.json", "w"))
