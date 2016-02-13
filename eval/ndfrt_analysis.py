from __future__ import division
import argparse
import numpy as np
import scipy as sp
from scipy.spatial.distance import cosine
from scipy.spatial.distance import cdist
from scipy.stats import pearsonr
from scipy.stats import spearmanr
from multiprocessing import Process, Queue

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


# MRCONSO.RRF is a file that needs to be downloaded from the UMLS Metathesaurus
def get_CUI_to_description():
    cui_to_description = {} 
    with open('MRCONSO.RRF', 'r') as infile:
        lines = infile.readlines()
        for row in lines:
            datum = row.strip().split('|')
            if datum[0] not in cui_to_description:
                cui_to_description[datum[0]] = datum[14]
    return cui_to_description

def get_CUI_to_type():
    CUI_to_type_hdr = 'MRSTY.RRF'
    CUI_to_type_map = {}
    with open(CUI_to_type_hdr, 'r') as infile:
        lines = infile.readlines()
        for line in lines:
            CUI = line.split('|')[0]
            type = line.split('|')[3]
            if CUI not in CUI_to_type_map:
                CUI_to_type_map[CUI] = [type]     
            else:
                CUI_to_type_map[CUI].append(type)
    return CUI_to_type_map


def read_embedding_matrix(filename):
    concept_to_cui, cui_to_concept = get_cui_concept_mappings() # comment out this after fix input
    with open(filename, 'r') as infile:
        embedding_num, dimension = map(int, infile.readline().strip().split(' '))
        # -1 for remove </s>
        embedding_matrix = np.zeros((embedding_num-1, dimension))
        data = infile.readlines() 
        idx_to_cui = {}
        cui_to_idx = {}
        for idx in xrange(embedding_num-1):
            datum = data[idx+1].strip().split(' ')
            cui = datum[0]
            if cui[0] != 'C':
                if cui in concept_to_cui:
                    cui = concept_to_cui[cui]
            embedding_matrix[idx,:] = np.array(map(float, datum[1:]))
            idx_to_cui[idx] = cui
            cui_to_idx[cui] = idx
        return embedding_matrix, idx_to_cui, cui_to_idx


def generate_overlapping_sets(filenames):
    embedding_idx_cui = {} # a dictionary of (embedding_matrix, idx_to_cui, cui_to_idx)
    overlapping_cuis = set([])

    if len(filenames) == 1:
        embedding_matrix, idx_to_cui, cui_to_idx = read_embedding_matrix(filenames[0])
        filename_to_embedding_matrix = {}
        filename_to_embedding_matrix[filenames[0]] = embedding_matrix
        return filename_to_embedding_matrix, idx_to_cui, cui_to_idx

    for fileidx, filename in enumerate(filenames):
        embedding_matrix, idx_to_cui, cui_to_idx = read_embedding_matrix(filename)
        embedding_idx_cui[filename] = (embedding_matrix, idx_to_cui, cui_to_idx)
        if fileidx == 0:
            overlapping_cuis.update(set(cui_to_idx.keys()))
        else:
            overlapping_cuis.intersection_update(set(cui_to_idx.keys()))
    overlapping_cuis = list(overlapping_cuis)

    idx_of_overlapping_cuis = {}
    for filename in filenames:
        idx_of_overlapping_cuis[filename] = []

    idx_to_cui = {}
    cui_to_idx = {}
    for idx, cui in enumerate(overlapping_cuis):
        idx_to_cui[idx] = cui
        cui_to_idx[cui] = idx
        for filename in filenames:
            idx_of_overlapping_cuis[filename].append(embedding_idx_cui[filename][2][cui])
    filename_to_embedding_matrix = {}
    for filename in filenames:
        idx_of_overlapping_cuis[filename] = np.array(idx_of_overlapping_cuis[filename])
        filename_to_embedding_matrix[filename] = embedding_idx_cui[filename][0][idx_of_overlapping_cuis[filename]]
    return filename_to_embedding_matrix, idx_to_cui, cui_to_idx


'''
def generate_overlapping_sets(filename1, filename2):
    embedding_matrix_1, idx_to_cui_1, cui_to_idx_1 = read_embedding_matrix(filename1)
    embedding_matrix_2, idx_to_cui_2, cui_to_idx_2 = read_embedding_matrix(filename2)
    overlapping_cuis = list(set(cui_to_idx_1.keys()).intersection(set(cui_to_idx_2.keys())))
    print 'set %s has %d concepts' %(filename1, embedding_matrix_1.shape[0])
    print 'set %s has %d concepts' %(filename2, embedding_matrix_2.shape[0])
    print 'the size of overlapping concept is %d\n' %(len(overlapping_cuis))
    idx_of_overlapping_cuis_1 = []
    idx_of_overlapping_cuis_2 = []
    idx_to_cui = {}
    cui_to_idx = {}
    for idx, cui in enumerate(overlapping_cuis):
        idx_of_overlapping_cuis_1.append(cui_to_idx_1[cui])
        idx_of_overlapping_cuis_2.append(cui_to_idx_2[cui])
        idx_to_cui[idx] = cui
        cui_to_idx[cui] = idx
    embedding_matrix_1_new = embedding_matrix_1[np.array(idx_of_overlapping_cuis_1)]
    embedding_matrix_2_new = embedding_matrix_2[np.array(idx_of_overlapping_cuis_2)]
    return embedding_matrix_1_new, embedding_matrix_2_new, idx_to_cui, cui_to_idx
'''


def get_neighbors(query_idx, target_indices, embedding_matrix, num_of_neighbor):
    vector = np.reshape(embedding_matrix[query_idx], (1, -1))
    Y = cdist(vector, embedding_matrix, 'cosine')
    ranks = np.argsort(Y)
    return ranks[0, :num_of_neighbor+1]


def get_analogy(ref_idx, seed_idx, query_idx, target_indices, embedding_matrix, num_of_neighbor):
    vector = embedding_matrix[seed_idx, :] - embedding_matrix[ref_idx, :] + embedding_matrix[query_idx, :]
    vector = np.reshape(vector, (1, -1))
    Y = cdist(vector, embedding_matrix, 'cosine')
    ranks = np.argsort(Y)
    return ranks[0, :num_of_neighbor+1]


def organize_cui_by_type(cui_to_idx):
    cui_to_type = get_CUI_to_type()
    type_to_idx = {}
    idx_to_type = {}
    for cui in cui_to_idx.keys():
        if cui in cui_to_type:
            types = cui_to_type[cui]
            idx_to_type[cui_to_idx[cui]] = types
            for type in types:
                if type in type_to_idx:
                    type_to_idx[type].add((cui_to_idx[cui]))
                else:
                    type_to_idx[type] = set([cui_to_idx[cui]])
    return type_to_idx, idx_to_type


def get_nn_analysis(cui_to_idx, embedding_matrix, num_of_neighbor, f):
    type_to_idx, idx_to_type = organize_cui_by_type(cui_to_idx)
    Y = cdist(embedding_matrix, embedding_matrix, 'cosine')
    ranks = np.argsort(Y)
    type_idx_dcg_err = {}
    type_dcg = {}
    type_err = {}
    print 'done calcualting distance'
    for type in type_to_idx.keys():
        type_dcg[type] = []
        type_err[type] = []
        type_idx_dcg_err[type] = []
    for idx in idx_to_type.keys():
        target = ranks[idx, 1:num_of_neighbor+1]
        for type in idx_to_type[idx]:
            dcg = 0
            err = 0 
            for i in xrange(num_of_neighbor):
                if target[i] in type_to_idx[type]:
                    dcg += np.reciprocal(np.log2(i+2))
                    if err == 0:
                        err = 1/(1+i)
            type_idx_dcg_err[type].append((idx, dcg, err))
            type_dcg[type].append(dcg)
            type_err[type].append(err)
    for type in type_to_idx.keys():
        print '%50s (DCG) %2.5f %2.5f' %(type, np.mean(np.array(type_dcg[type])), np.std(np.array(type_dcg[type])))
        print '%50s (ERR) %2.5f %2.5f' %(type, np.mean(np.array(type_err[type])), np.std(np.array(type_err[type])))
        f.write('%50s (DCG) %2.5f %2.5f\n' %(type, np.mean(np.array(type_dcg[type])), np.std(np.array(type_dcg[type]))))
        f.write('%50s (ERR) %2.5f %2.5f\n' %(type, np.mean(np.array(type_err[type])), np.std(np.array(type_err[type]))))
    return type_idx_dcg_err
            


def get_all_target_neighbors(query_to_targets, embedding_matrix, num_of_neighbor):
    vectors = embedding_matrix[np.array(query_to_targets.keys()), :]
    Y = cdist(vectors, embedding_matrix, 'cosine')
    ranks = np.argsort(Y)
    query_target_rank = {}
    for idx, query in enumerate(query_to_targets.keys()):
        targets_list = list(query_to_targets[query])
        target_rank = []
        for target in targets_list:
            target_rank.append(np.where(ranks[idx, :] == target)[0][0])
        query_target_rank[query] = (zip(targets_list, target_rank), ranks[idx, :num_of_neighbor+1])
    return query_target_rank


def get_all_target_analogies(ref_idx, seed_idx, query_to_targets, embedding_matrix, num_of_neighbor):
    ref_vecs = np.tile(embedding_matrix[seed_idx, :] - embedding_matrix[ref_idx], (len(query_to_targets), 1))
    vectors = ref_vecs + embedding_matrix[np.array(query_to_targets.keys()), :]
    Y = cdist(vectors, embedding_matrix, 'cosine')
    ranks = np.argsort(Y)
    query_target_rank = {}
    for idx, query in enumerate(query_to_targets.keys()):
        targets_list = list(query_to_targets[query])
        target_rank = []
        for target in targets_list:
            target_rank.append(np.where(ranks[idx, :] == target)[0][0])
        query_target_rank[query] = (zip(targets_list, target_rank), ranks[idx, :num_of_neighbor+1])
    return query_target_rank


def get_drug_diseases_to_check(concept_filename, cui_to_idx):
    query_to_targets = {}
    outfile = open('drug_disease_' + concept_filename.split('/')[-1] , 'w')
    cui_to_description = get_CUI_to_description()
    with open(concept_filename, 'r') as infile:
        data = infile.readlines()
        for row in data:
            drug, diseases = row.strip().split(':')
            diseases = diseases.split(',')[:-1]
            if drug in cui_to_idx and cui_to_idx[drug] not in query_to_targets:
                disease_set = set([])
                disease_cui_set = set([])
                for disease in diseases:
                    if disease in cui_to_idx:
                        disease_set.add(cui_to_idx[disease])
                        disease_cui_set.add(disease)
                if len(disease_set) > 0:
                    outfile.write('%s(%s):' %(drug, cui_to_description[drug]))
                    for cui in disease_cui_set:
                        outfile.write('%s(%s),' %(cui, cui_to_description[cui]))
                    outfile.write('\n')
                    query_to_targets[cui_to_idx[drug]] = disease_set
    outfile.close()
    print '%d/%d concepts are found in embeddings' %(len(query_to_targets), len(data))
    return query_to_targets



def get_drug_pairs_to_check(concept_filename, cui_to_idx):
    drug_pairs = {}
    query_to_targets_cui = {}
    with open(concept_filename, 'r') as infile:
        data = infile.readlines()
        disease_to_drugs = {}
        for row in data:
            drug, diseases = row.strip().split(':')
            diseases = diseases.split(',')[:-1]
            query_to_targets_cui[drug] = set(diseases)
            for idx, disease in enumerate(diseases):
                if disease not in disease_to_drugs:
                    disease_to_drugs[disease] = []
                if drug in cui_to_idx:
                    disease_to_drugs[disease].append(cui_to_idx[drug])

        for disease in disease_to_drugs.keys():
            if len(disease_to_drugs[disease]) > 1:
                for idx, drug in enumerate(disease_to_drugs[disease]):
                    if drug not in drug_pairs:
                        drug_pairs[drug] = set([])
                    drug_pairs[drug].update(set(disease_to_drugs[disease][:idx]))
                    drug_pairs[drug].update(set(disease_to_drugs[disease][idx+1:]))
    return drug_pairs, query_to_targets_cui


def display_query_target_rank(query_target_rank, idx_to_cui, seed_pair=None):
    cui_to_description = get_CUI_to_description()
    CUI_to_type_map = get_CUI_to_type()
    for query in query_target_rank.keys():
        query_cui = idx_to_cui[query]
        query_name = cui_to_description[query_cui]
        if not seed_pair:
            print 'Neighbors for %9s %s' %(query_cui, query_name)
        else:
            ref_idx, seed_idx = seed_pair
            ref_cui = idx_to_cui[ref_idx]
            ref_name = cui_to_description[ref_cui]
            seed_cui = idx_to_cui[seed_idx]
            seed_name = cui_to_description[seed_cui]
            print 'Analogy for %s %s : %s %s = %s %s : ?' %(ref_cui, ref_name, seed_cui, seed_name, query_cui, query_name)
        print '------------------------------------------------------------'
        target_rank_pairs, top_neighbors = query_target_rank[query]
        for target_idx, rank in target_rank_pairs:
            target_cui = idx_to_cui[target_idx]
            target_name = cui_to_description[target_cui]
            print '%5d %9s %s' %(rank, target_cui, target_name),
            if target_cui in CUI_to_type_map:
                print CUI_to_type_map[target_cui]
            else:
                print ""
        for index, target_idx in enumerate(list(top_neighbors)):
            target_cui = idx_to_cui[target_idx]
            if target_cui not in cui_to_description:
                cui_to_description[target_cui] = target_cui
            target_name = cui_to_description[target_cui]
            print '%5d %9s %s' %(index, target_cui, target_name),
            if target_cui in CUI_to_type_map:
                print CUI_to_type_map[target_cui]
            else:
                print ""
        print ""


def evaluate_result(query_target_rank, num_of_nn):
    num_of_queries = len(query_target_rank)
    num_of_hits = 0
    for query in query_target_rank.keys():
        target_rank_pairs, top_neighbors = query_target_rank[query]
        for target_idx, rank in target_rank_pairs:
            if rank <= num_of_nn:
                num_of_hits += 1
                break
    #print '%5d out of %5d queries (%2.4f)' %(num_of_hits, num_of_queries, (num_of_hits*100)/num_of_queries)
    #f.write('%5d out of %5d queries (%2.4f)\n' %(num_of_hits, num_of_queries, (num_of_hits*100)/num_of_queries))
    return num_of_hits


def analyze_semantic_files_child(result_q, pidx, n1, n2, ref_seed_list, query_to_targets, embedding_matrix, num_of_nn):
    counter = 0
    ref_seed_hit_list = []
    hit_sum = 0
    hit_max = (-1, -1, 0)
    for idx in xrange(n1, n2):
        counter += 1
        #if (idx-n1) % 10 == 0:
        #    print pidx, idx-n1
        ref_idx, seed_idx = ref_seed_list[idx]
        query_target_rank = get_all_target_analogies(ref_idx, 
                                                     seed_idx, 
                                                     query_to_targets, 
                                                     embedding_matrix, 
                                                     num_of_nn)
        num_of_hits = evaluate_result(query_target_rank, num_of_nn)
        hit_sum += num_of_hits
        if num_of_hits > hit_max[2]:
            hit_max = (ref_idx, seed_idx, num_of_hits)
        ref_seed_hit_list.append((ref_idx, seed_idx, num_of_hits))
    result_q.put((counter, hit_sum, hit_max))


def analyze_semantic_files(filenames, num_of_nn, concept_file, num_of_cores):
    filename_to_embedding_matrices, idx_to_cui, cui_to_idx = generate_overlapping_sets(filenames)
    print len(idx_to_cui)
    query_to_targets = get_drug_diseases_to_check(concept_file, cui_to_idx)
    all_queries = query_to_targets.keys()
   
    fname = 'analysis_semantic_' + concept_file.split('/')[-1].split('.')[0] + '.txt'
    f = open(fname, 'w')
    #print f

    num_of_queries = len(all_queries)
    f.write('number of queries: %d\n' %(num_of_queries))
    
    cui_to_description = get_CUI_to_description()

    for filename in filenames:
        query_target_rank = get_all_target_neighbors(query_to_targets, filename_to_embedding_matrices[filename], num_of_nn)
        num_of_hits = evaluate_result(query_target_rank, num_of_nn)
        print '%s &  %.2f,' %(filename.split('/')[-1],
                num_of_hits*100/num_of_queries),
        f.write('%s,%.4f,%d,' %(filename.split('/')[-1], num_of_hits*100/num_of_queries, num_of_hits))
        
        ref_seed_list = []
        for ref_idx in all_queries:
            for seed_idx in query_to_targets[ref_idx]:
                ref_seed_list.append((ref_idx, seed_idx))

        result_q = Queue()
        process_list = []
        N = len(ref_seed_list)
        #print N
        chunk_size = np.ceil(N/num_of_cores)

        for i in xrange(num_of_cores):
            n1 = min(int(i*chunk_size), N)
            n2 = min(int((i+1)*chunk_size), N)
            p = Process(target=analyze_semantic_files_child, 
                        args=(result_q, i, n1, n2, 
                              ref_seed_list,
                              query_to_targets,
                              filename_to_embedding_matrices[filename],
                              num_of_nn))
            process_list.append(p)

        for p in process_list:
            p.start()

        for p in process_list:
            p.join()

        counter = 0
        hit_sum = 0
        hit_max = (-1, -1, 0)
        for p in process_list:
            counter_part, hit_sum_part, hit_max_part = result_q.get()
            counter += counter_part
            hit_sum += hit_sum_part
            if hit_max_part[2] > hit_max[2]:
                hit_max = hit_max_part

        ref_cui = idx_to_cui[hit_max[0]]
        ref_name = cui_to_description[ref_cui]
        seed_cui = idx_to_cui[hit_max[1]]
        seed_name = cui_to_description[seed_cui]
        print '& %.2f & %.2f  \\\\' %(hit_sum/counter*100/num_of_queries,
                                          hit_max[2]*100/num_of_queries)
        f.write('%.4f,%.4f,%s,%s,%.4f,%d\n' %(hit_sum/counter*100/num_of_queries,
                                              hit_sum/counter,
                                              ref_name, seed_name,
                                              hit_max[2]*100/num_of_queries,
                                              hit_max[2]))
    f.close()


def analyze_semantic(filename1, filename2, num_of_nn, concept_file):
    embedding_matrix_1, embedding_matrix_2, idx_to_cui, cui_to_idx = generate_overlapping_sets(filename1, filename2)
    query_to_targets = get_drug_diseases_to_check(concept_file, cui_to_idx)
    all_queries = query_to_targets.keys()

    fname = 'new_result/' + filename2.split('/')[-1].split('.')[0] + "_" + concept_file.split('/')[-1].split('.')[0]
    f = open(fname, 'w')
    f.write('%s\n%s\n%s\n' % (filename1, filename2, concept_file))

    #print "\nNeighbor result of de Vine et al."
    f.write("\nNeighbor result of de Vine et al.\n")
    query_target_rank_1_n = get_all_target_neighbors(query_to_targets, embedding_matrix_1, num_of_nn)
    evaluate_result(query_target_rank_1_n, num_of_nn, f)
    #display_query_target_rank(query_target_rank_1_n, idx_to_cui)
    
    #print "\nNeighbor result of Stanford"
    f.write("\nNeighbor result of Stanford\n")
    query_target_rank_2_n = get_all_target_neighbors(query_to_targets, embedding_matrix_2, num_of_nn)
    evaluate_result(query_target_rank_2_n, num_of_nn, f)
    #display_query_target_rank(query_target_rank_2_n, idx_to_cui)
    
    cui_to_description = get_CUI_to_description()
    for ref_idx in all_queries:
        for seed_idx in list(query_to_targets[ref_idx]):
            ref_cui = idx_to_cui[ref_idx]
            ref_name = cui_to_description[ref_cui]
            seed_cui = idx_to_cui[seed_idx]
            seed_name = cui_to_description[seed_cui]
            #print '\nAnalogy using seed %s %s : %s %s' %(ref_cui, ref_name, seed_cui, seed_name)
            f.write('\nAnalogy using seed %s %s : %s %s\n' %(ref_cui, ref_name, seed_cui, seed_name))

            #print 'de Vine'
            f.write('de Vine\n')
            query_target_rank_1_a = get_all_target_analogies(ref_idx, seed_idx, query_to_targets, embedding_matrix_1, num_of_nn)
            evaluate_result(query_target_rank_1_a, num_of_nn, f)
            #display_query_target_rank(query_target_rank_1_a, idx_to_cui, (ref_idx, seed_idx))
            
            #print 'Stanford'
            f.write('Stanford\n')
            query_target_rank_2_a = get_all_target_analogies(ref_idx, seed_idx, query_to_targets, embedding_matrix_2, num_of_nn)
            evaluate_result(query_target_rank_2_a, num_of_nn, f)
            #display_query_target_rank(query_target_rank_2_a, idx_to_cui, (ref_idx, seed_idx))
    f.close()


def get_fine_grain_drug(idx_to_cui, embedding_matrix, drug_pairs, search_indices, query_to_targets_cui, num_of_neighbor, display=False):
    cui_to_description = get_CUI_to_description()

    query_indices = np.array(drug_pairs.keys())
    Y = cdist(embedding_matrix[query_indices, :], embedding_matrix[search_indices, :], 'cosine')
    ranks = np.argsort(Y)
    cumulative_ndcgs = []
    for counter, query_idx in enumerate(list(query_indices)):
        target = ranks[counter, 1:num_of_neighbor+1]
        num_of_possible_hits = min(len(drug_pairs[query_idx]), num_of_neighbor)
        if display:
            print ""
            print cui_to_description[idx_to_cui[query_idx]] + ' (' + str(num_of_possible_hits) + '): ',
            for disease_cui in query_to_targets_cui[idx_to_cui[query_idx]]:
                print disease_cui, 
                if disease_cui in cui_to_description:
                    print '(' + cui_to_description[disease_cui] + '),',

            print ''
            print '-------------------------------------------'
        dcg = 0
        best_dcg = np.sum(np.reciprocal(np.log2(range(2, num_of_possible_hits+2))))
        for i in xrange(num_of_neighbor):
            if search_indices[target[i]] in drug_pairs[query_idx]:
                dcg += np.reciprocal(np.log2(i+2))
                if display:
                    print 'hit: ',
            else:
                if display:
                    print '     ',
            if display:
                print cui_to_description[idx_to_cui[search_indices[target[i]]]] + ': ', 
                if idx_to_cui[search_indices[target[i]]] in query_to_targets_cui:
                    for disease_cui in query_to_targets_cui[idx_to_cui[search_indices[target[i]]]]:
                        print disease_cui,
                        if disease_cui in cui_to_description:
                            print '(' + cui_to_description[disease_cui] + '),',
                print ''
        cumulative_ndcgs.append(dcg/best_dcg)
        if display:
            print dcg/best_dcg
    #print ''    
    #print np.mean(np.array(cumulative_ndcgs))
    #print np.median(np.array(cumulative_ndcgs))
    #print ''
    return cumulative_ndcgs, np.mean(np.array(cumulative_ndcgs))


def analyze_fine_grain_concept_similarity_files(filenames, concept_filename, num_of_nn):
    filename_to_embedding_matrices, idx_to_cui, cui_to_idx = generate_overlapping_sets(filenames)

    drug_pairs, query_to_targets_cui = get_drug_pairs_to_check(concept_filename, cui_to_idx)
    drug_to_check = set([])
    for drug in drug_pairs.keys():
        drug_to_check.add(idx_to_cui[drug])
    
    print drug_to_check.difference(set(query_to_targets_cui))
    print len(query_to_targets_cui)
    print len(drug_pairs)

    type_to_idx, idx_to_type = organize_cui_by_type(cui_to_idx)
    cui_to_description = get_CUI_to_description()
    
    '''
    search_indices = set([])
    search_indices.update(set(drug_pairs.keys()))
    search_indices.update(type_to_idx['Pharmacologic Substance'])
    search_indices.update(type_to_idx['Antibiotic'])
    search_indices = np.array(list(search_indices))
    '''
    search_indices = np.array(drug_pairs.keys())
    for filename in filenames:
        cumulative_ndcgs, mean_ndcgs = get_fine_grain_drug(idx_to_cui, filename_to_embedding_matrices[filename], drug_pairs, search_indices, query_to_targets_cui, num_of_nn)
        print filename + ' & ' + str(mean_ndcgs) + '  \\\\ '


def analyze_concept_similarity_files(filenames, num_of_nn):
    filename_to_embedding_matrices, idx_to_cui, cui_to_idx = generate_overlapping_sets(filenames)

    fname = 'analysis_concept_similarity'
    f = open(fname, 'w')
    print f

    for filename in filenames:
        f.write('%s\n' %(fname))
        type_idx_dcg_err = get_nn_analysis(cui_to_idx, filename_to_embedding_matrices[filename], num_of_nn, f)
        f.write('\n')
    
    f.close()


# compute neighbor score for UMLS
def analyze_concept_similarity(filename1, filename2, num_of_nn):
    embedding_matrix_1, embedding_matrix_2, idx_to_cui, cui_to_idx = generate_overlapping_sets(filename1, filename2)
    print 'result for de Vine'
    type_idx_dcg_err_1 = get_nn_analysis(cui_to_idx, embedding_matrix_1, num_of_nn)
    print '\nresult for Stanford'
    type_idx_dcg_err_2 = get_nn_analysis(cui_to_idx, embedding_matrix_2, num_of_nn)


def analyze_concept_relatedness(filenames):
    pairs_to_evaluate = []
    cui_to_description = {}
    with open('caviedes.tsv', 'r') as infile:
        data = infile.readlines()
        for row in data:
            elements = row.strip().split('\t')
            pairs_to_evaluate.append((elements[1], elements[3], float(elements[4])))
            cui_to_description[elements[1]] = elements[0]
            cui_to_description[elements[3]] = elements[2]

    filename_to_embedding_matrices, idx_to_cui, cui_to_idx = generate_overlapping_sets(filenames)

    caviedes = []
    caviedes_to_print = []
    for cui1, cui2, similarity_score in pairs_to_evaluate:
        if cui1 in cui_to_idx and cui2 in cui_to_idx:
            caviedes.append(similarity_score)
            caviedes_to_print.append((cui_to_description[cui1], 
                                      cui_to_description[cui2], 
                                      similarity_score))
    for val in sorted(caviedes_to_print, key=lambda ele: ele[2]):
        print val

    for filename in filenames:
        embedding_matrix = filename_to_embedding_matrices[filename]
        Y = cdist(embedding_matrix, embedding_matrix, 'cosine')
        print filename
        Y_scaled = Y/Y.max()
        current = []
        current_to_print = []
        for cui1, cui2, similarity_score in pairs_to_evaluate:
            if cui1 in cui_to_idx and cui2 in cui_to_idx:
                print cui1, cui2
                current.append(Y_scaled[cui_to_idx[cui1], cui_to_idx[cui2]]) 
                current_to_print.append((cui_to_description[cui1],
                                         cui_to_description[cui2],
                                         Y_scaled[cui_to_idx[cui1], cui_to_idx[cui2]]))
            else:
                print 'not a hit: ', cui1, cui2
        caviedes = np.array(caviedes)
        current = np.array(current)
        print pearsonr(caviedes, current)
        print spearmanr(caviedes, current)
        for val in sorted(current_to_print, key=lambda ele: ele[2]):
            print val



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis_type", type=int, default=1, help='1:ndt-rf, 2:')
    parser.add_argument("--filenames", default='files_to_process.txt')
    parser.add_argument("--number_of_nn", type=int, default=40, help='the number of neighbors to show')
    parser.add_argument("--number_of_cores", type=int, default=60, help='the number of cores to use')
    parser.add_argument("--concept_file", default='ndf-rt/may_treat_cui.txt', help='input the concept file for query')
    args = parser.parse_args()
    analysis_type = args.analysis_type
    filenames_file = args.filenames
    num_of_nn = args.number_of_nn
    num_of_cores = args.number_of_cores
    concept_file = args.concept_file
    filenames = []
    with open(filenames_file, 'r') as infile:
        data = infile.readlines()
        for row in data:
            filenames.append(row.strip())
    analyze_semantic_files(filenames, num_of_nn, concept_file, num_of_cores)
    #analyze_concept_similarity_files(filenames, num_of_nn)

    # experiment for table 3
    #analyze_fine_grain_concept_similarity_files(filenames, concept_file, num_of_nn)
    
    #analyze_concept_relatedness(filenames)
    #if analysis_type ==  1:
    #    analyze_semantic(filename1, filename2, num_of_nn, concept_file)
    #elif analysis_type == 2:
    #    analyze_concept_similarity(filename1, filename2, num_of_nn)

