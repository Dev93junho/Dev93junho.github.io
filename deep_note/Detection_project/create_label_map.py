import json

f = json.load(open('cards_coco_val.json'))
dst = open('label_map.pbtxt', 'w')

for cat in f['categories']:
    dst.write('item {\n')
    dst.write('  id: %d\n' % cat['id'])
    dst.write('  name: \'%s\'\n' % cat['name'])
    dst.write('}\n')
dst.close()