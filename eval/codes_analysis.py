from __future__ import division
import argparse
import numpy as np
import scipy as sp
from scipy.spatial.distance import cdist
from icd9 import ICD9

tree = ICD9('codes.json')

def get_icd9_pairs(icd9_set):
    icd9_pairs = {}
    with open('icd9_grp_file.txt', 'r') as infile:
        data = infile.readlines()
        for row in data:
            codes, name = row.strip().split('#')
            name = name.strip()
            codes = codes.strip().split(' ')
            new_codes = set([])
            for code in codes:
                if code in icd9_set:
                    new_codes.add(code)
                elif len(code) > 5 and code[:5] in icd9_set:
                    new_codes.add(code[:5])
                elif len(code) > 4 and code[:3] in icd9_set:
                    new_codes.add(code[:3])
            codes = list(new_codes)

            if len(codes) > 1:
                for idx, code in enumerate(codes):
                    if code not in icd9_pairs:
                        icd9_pairs[code] = set([])
                    icd9_pairs[code].update(set(codes[:idx]))
                    icd9_pairs[code].update(set(codes[idx+1:]))
    return icd9_pairs


def get_coarse_icd9_pairs(icd9_set): 
    icd9_pairs = {}
    ccs_to_icd9 = {}
    with open('ccs_coarsest.txt', 'r') as infile:
        data = infile.readlines()
        currect_ccs = ''
        for row in data:
            if row[:10].strip() != '':
                current_ccs = row[:10].strip()
                ccs_to_icd9[current_ccs] = set([])
            elif row.strip() != '':
                ccs_to_icd9[current_ccs].update(set(row.strip().split(' ')))

    ccs_coarse = {}
    for ccs in ccs_to_icd9.keys():
        ccs_eles = ccs.split('.')
        if len(ccs_eles) >= 2:
            code = ccs_eles[0] + '.' + ccs_eles[1]
            if code not in ccs_coarse:
                ccs_coarse[code] = set([])
            ccs_coarse[code].update(ccs_to_icd9[ccs])
    
    for ccs in ccs_coarse.keys():
        new_codes = set([])
        for code in ccs_coarse[ccs]:
            if len(code) > 3:
                new_code = code[:3] + '.' + code[3:]
            code = new_code
            if code in icd9_set:
                new_codes.add(code)
            elif len(code) > 5 and code[:5] in icd9_set:
                new_codes.add(code[:5])
            elif len(code) > 4 and code[:3] in icd9_set:
                new_codes.add(code[:3])
        codes = list(new_codes)
        if len(codes) > 1:
            for idx, code in enumerate(codes):
                if code not in icd9_pairs:
                    icd9_pairs[code] = set([])
                icd9_pairs[code].update(set(codes[:idx]))
                icd9_pairs[code].update(set(codes[idx+1:]))
    return icd9_pairs

def get_cui_concept_mappings():
    concept_to_cui_hdr = '2b_concept_ID_to_CUI.txt'
    concept_to_cui = {}
    cui_to_concept = {}
    with open(concept_to_cui_hdr, 'r') as infile:
        lines = infile.readlines()
        for line in lines:
            concept = line.split('\t')[0]
            cui = line.split('\t')[1].split('\r')[0]
            concept_to_cui[concept] = cui 
            cui_to_concept[cui] = concept
    return concept_to_cui, cui_to_concept


def get_icd9_cui_mappings():
    cui_to_icd9 = {}
    icd9_to_cui = {}
    with open('cui_icd9.txt', 'r') as infile:
        data = infile.readlines()
        for row in data:
            ele = row.strip().split('|')
            if ele[11] == 'ICD9CM':
                cui = ele[0]
                icd9 = ele[10]
                if cui not in cui_to_icd9 and icd9 != '' and '-' not in icd9:
                    cui_to_icd9[cui] = icd9
                    icd9_to_cui[icd9] = cui
    return cui_to_icd9, icd9_to_cui


def read_embedding_cui(filename):
    concept_to_cui, cui_to_concept = get_cui_concept_mappings() # comment out this after fix input
    cui_to_icd9, icd9_to_cui = get_icd9_cui_mappings()
    with open(filename, 'r') as infile:
        embedding_num, dimension = map(int, infile.readline().strip().split(' '))
        # -1 for remove </s>
        embedding_matrix = np.zeros((embedding_num-1, dimension))
        data = infile.readlines() 
        idx_to_name = {}
        name_to_idx = {}
        embedding_type_to_indices = {}
        embedding_type_to_indices['IDX'] = []
        embedding_type_to_indices['O'] = []
        for idx in xrange(embedding_num-1):
            datum = data[idx+1].strip().split(' ')
            cui = datum[0]
            if cui[0] != 'C':
                if cui in concept_to_cui:
                    cui = concept_to_cui[cui]
            embedding_matrix[idx,:] = np.array(map(float, datum[1:]))
            # potential bug here
            if cui in cui_to_icd9:
                idx_to_name[idx] = cui_to_icd9[cui]
                name_to_idx[cui_to_icd9[cui]] = idx
                embedding_type_to_indices['IDX'].append(idx)
            else:
                idx_to_name[idx] = cui
                name_to_idx[cui] = idx
                embedding_type_to_indices['O'].append(idx)
        return embedding_matrix, embedding_type_to_indices, name_to_idx, idx_to_name
    

def read_embedding_codes(filename):
    with open(filename, 'r') as infile:
        embedding_num, dimension = map(int, infile.readline().strip().split(' '))
        # -1 for remove </s>
        embedding_matrix = np.zeros((embedding_num-1, dimension))
        data = infile.readlines()
        embedding_type_to_indices = {}
        name_to_idx = {}
        idx_to_name = {}
        for idx in xrange(embedding_num-1):
            datum = data[idx+1].strip().split(' ')
            embedding_name = datum[0]
            embedding_type, embedding_value = embedding_name.split('_')
            name_to_idx[embedding_value] = idx
            idx_to_name[idx] = embedding_value 
            if embedding_type not in embedding_type_to_indices:
                embedding_type_to_indices[embedding_type] = []
            embedding_type_to_indices[embedding_type].append(idx)
            embedding_matrix[idx,:] = np.array(map(float, datum[1:]))
        return embedding_matrix, embedding_type_to_indices, name_to_idx, idx_to_name


def generate_overlapping_sets(filenames_type):
    embedding_idx_icd9 = {} # a dictionary of (embedding_matrix, idx_to_icd9, icd9_to_idx)
    overlapping_icd9s = set([])
    start = 1

    for filename, embedding_type in filenames_type:
        #print filename
        #print embedding_type
        if embedding_type == 'codes':
            embedding_matrix, embedding_type_to_indices, icd9_to_idx, idx_to_icd9 = read_embedding_codes(filename)
            embedding_idx_icd9[filename] = (embedding_matrix, idx_to_icd9, icd9_to_idx)
            if start == 1:
                start = 0
                overlapping_icd9s.update(set(icd9_to_idx.keys()))
            else:
                overlapping_icd9s.intersection_update(set(icd9_to_idx.keys()))
                #print len(overlapping_icd9s)
        elif embedding_type == 'cui':
            embedding_matrix, embedding_type_to_indices, icd9_to_idx, idx_to_icd9 = read_embedding_cui(filename)
            embedding_idx_icd9[filename] = (embedding_matrix, idx_to_icd9, icd9_to_idx)
            if start == 1:
                start = 0
                overlapping_icd9s.update(set(icd9_to_idx.keys()))
            else:
                overlapping_icd9s.intersection_update(set(icd9_to_idx.keys()))
                #print len(overlapping_icd9s)
    overlapping_icd9s = list(overlapping_icd9s)

    idx_of_overlapping_icd9s = {}
    for filename, embedding_type in filenames_type:
        idx_of_overlapping_icd9s[filename] = []

    idx_to_icd9 = {}
    icd9_to_idx = {}
    for idx, icd9 in enumerate(overlapping_icd9s):
        idx_to_icd9[idx] = icd9
        icd9_to_idx[icd9] = idx
        for filename, embedding_type in filenames_type:
            idx_of_overlapping_icd9s[filename].append(embedding_idx_icd9[filename][2][icd9])

    filename_to_embedding_matrix = {}
    for filename, embedding_type in filenames_type:
        idx_of_overlapping_icd9s[filename] = np.array(idx_of_overlapping_icd9s[filename])
        filename_to_embedding_matrix[filename] = embedding_idx_icd9[filename][0][idx_of_overlapping_icd9s[filename]]
    return filename_to_embedding_matrix, idx_to_icd9, icd9_to_idx


def get_icd9_to_description():
    icd9_to_description = {}
    with open('CMS32_DESC_LONG_DX.txt', 'r') as infile:
        data = infile.readlines()
        for row in data:
            icd9 = row.strip()[:6].strip()
            if len(icd9) > 3:
                icd9 = icd9[:3] + '.' + icd9[3:]
            description = row.strip()[6:].strip()
            icd9_to_description[icd9] = description
    return icd9_to_description


# type == f: fine grain, c: coarse grain
def get_css_analysis(filenames_type, num_of_neighbor, type='f'):
    filename_to_embedding_matrix, idx_to_icd9, icd9_to_idx = generate_overlapping_sets(filenames_type)
    print len(icd9_to_idx.keys())
    if type == 'c':
        icd9_pairs = get_coarse_icd9_pairs(set(icd9_to_idx.keys()))
    else:
        icd9_pairs = get_icd9_pairs(set(icd9_to_idx.keys()))

    print len(icd9_pairs)
    icd9_to_check = set(icd9_pairs.keys())
    icd9_to_check.intersection_update(set(icd9_to_idx.keys()))

    print len(icd9_to_check)

    icd9_to_description = get_icd9_to_description()
    for icd9 in icd9_to_idx.keys():
        if icd9 not in icd9_to_description:
            if tree.find(icd9):
                icd9_to_description[icd9] = tree.find(icd9).description.encode('utf-8')
            else:
                icd9_to_description[icd9] = ''


    filename_all = []
    value_all = []
    for filename, embedding_type in filenames_type:
        #print filename
        icd9_embeddings = filename_to_embedding_matrix[filename]
        Y = cdist(icd9_embeddings, icd9_embeddings, 'cosine')
        ranks = np.argsort(Y)
    
        cumulative_ndcgs = []

        for icd9 in icd9_to_check:
            target = ranks[icd9_to_idx[icd9], 1:num_of_neighbor+1]
            num_of_possible_hits = 0
            
            icd9_to_remove = set()

            for val in icd9_pairs[icd9]:
                if val not in icd9_to_idx:
                    icd9_to_remove.add(val)
            icd9_pairs[icd9].difference(icd9_to_remove)

            num_of_possible_hits = min(len(icd9_pairs[icd9]), num_of_neighbor)
            #print icd9 + '(' + str(num_of_possible_hits) + ')',
            #if icd9 in icd9_to_description:
            #    print '(' + icd9_to_description[icd9] + ')',
            #print ''
            #print '-------------------------------------------'
            dcg = 0
            best_dcg = np.sum(np.reciprocal(np.log2(range(2, num_of_possible_hits+2))))
            for i in xrange(num_of_neighbor):
                if idx_to_icd9[target[i]] in icd9_pairs[icd9]:
                    dcg += np.reciprocal(np.log2(i+2))
                    #print 'hit: ',
                #else:
                    #print '     ',
                #print idx_to_icd9[target[i]],
                #if idx_to_icd9[target[i]] in icd9_to_description:
                    #print icd9_to_description[idx_to_icd9[target[i]]],
                #print ''
            #print dcg/best_dcg
            #print ''
            cumulative_ndcgs.append(dcg/best_dcg)
        filename_all.append((filename))
        value_all.append(np.mean(np.array(cumulative_ndcgs)))
    return filename_all, value_all


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--filenames", default='orig_files_all.txt')
    args = parser.parse_args()
    filenames_file = args.filenames

    filenames = []
    with open(filenames_file, 'r') as infile:
        data = infile.readlines()
        for row in data:
            filenames.append(row.strip().split(','))
    
    get_css_analysis(filenames, 40, 'f')
