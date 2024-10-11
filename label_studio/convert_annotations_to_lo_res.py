import json
import os


fname = '/home/luke/Downloads/annotations_orig.json'
os.path.exists(fname)

with open(fname, 'r') as j:
    aa = json.load(j)

# Extracted from SQLlite db, and edited to contain filenames only
fname_new_tasks = '/home/luke/mydata/task_dump.csv'
os.path.exists(fname_new_tasks)
with open(fname_new_tasks, 'r') as f:
    task_imgs = f.readlines()
    task_imgs = [l.strip() for l in task_imgs]


aa_new = []
for i, a in enumerate(aa):
    my_annotation = {
        'data': {'img': task_imgs[i]},
        'predictions': [{
            'score': 1,
            'model_version': 'project1',
        }]
    }
    old_annotation = a['annotations'][0]
    my_annotation['predictions'][0]['task'] = old_annotation['task']

    my_results = []
    for result in old_annotation['result']:
        my_result = {}
        # Adjust result values where necessary. Note that label locations are recorded as %s rel to the image dimensions
        for k, v in result.items():
            if k == 'original_width':
                my_result[k] = 1024  # lower resolution (orig: 4032)
            elif k == 'original_height':
                my_result[k] = 768  # lower resolution (orig: 3024)
            else:
                my_result[k] = v

        my_results.append(my_result)

    my_annotation['predictions'][0]['result'] = my_results
    aa_new.append(my_annotation)


# save results in the predictions format
version = 'v4'
fname_save = f'/home/luke/Downloads/annotations_preds_{version}.json'
with open(fname_save, 'w') as j:
    json.dump(aa_new, j, indent=4)
