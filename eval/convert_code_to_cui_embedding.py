import numpy as np
from codes_analysis import get_icd9_cui_mappings

# using best
ingredient_to_best_ndc = {}
with open('ingredient_drug_ndc.txt', 'r') as infile:
    data = infile.readlines()
    for row in data:
        eles = row.strip().split(' ')
        ingredient_to_best_ndc[eles[0]] = eles[2]

# using parents with smallest number of child
ingredient_to_ndcs = {}
with open('ingredient_ndcs.txt', 'r') as infile:
    data = infile.readlines()
    for row in data:
        eles = row.strip().split(' ')
        ingredient_to_ndcs[eles[0]] = eles[1:]

# using all parents --- this is the method of converting NDCs to CUIs
# that we are using for this paper. The below are used in the paper.
ingredient_to_all_ndcs = {}
with open('ingredient_all_ndcs.txt', 'r') as infile:
    data = infile.readlines()
    for row in data:
        eles = row.strip().split(' ')
        ingredient_to_all_ndcs[eles[0]] = eles[1:]

embedding_filename = '../claims_codes_hs_300.txt'

cui_to_icd9, icd9_to_cui = get_icd9_cui_mappings()
ndc_to_embeddings = {}
icd9_cui_to_embeddings = {}
with open(embedding_filename, 'r') as infile:
    data = infile.readlines()
    for row in data:
        eles = row.strip().split(' ')
        name = eles[0]
        embedding = ' '.join(eles[1:])
        if name[0] == 'N':
            ndc_to_embeddings[name[2:]] = embedding
        if name[0] == 'I':
            if name[4:] in icd9_to_cui:
                cui = icd9_to_cui[name[4:]]
                icd9_cui_to_embeddings[cui] = embedding

ndc_embeddings = []
for ingredient in ingredient_to_all_ndcs.keys():
    ndcs = ingredient_to_all_ndcs[ingredient]
    embeddings = []
    for ndc in ndcs:
        if ndc in ndc_to_embeddings:
            embedding = map(float, ndc_to_embeddings[ndc].split(' '))
            embeddings.append(embedding)
    if len(embeddings) > 0:
        embeddings = np.array(embeddings)
        embedding = np.mean(embeddings, axis=0)
        ndc_embeddings.append((ingredient, embedding))
    else:
        print 'not found'

outfilename = '../claims_cuis_hs_300.txt'
with open(outfilename, 'w') as outfile:
    outfile.write('%s %s\n' %(len(ndc_embeddings) + len(icd9_cui_to_embeddings), embedding.shape[0]))
    for (ingredient, embedding) in ndc_embeddings:
        outfile.write('%s ' %(ingredient))
        for i in range(embedding.shape[0]):
            outfile.write('%.6f ' %(embedding[i]))
        outfile.write('\n')
    for cui in icd9_cui_to_embeddings:
        outfile.write('%s %s\n' %(cui, icd9_cui_to_embeddings[cui]))
