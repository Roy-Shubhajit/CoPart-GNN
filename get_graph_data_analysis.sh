#run graph_data_analysis.py for all datasets and coarsening ratios
for dataset in citeseer pubmed dblp Physics
do
    for coarsening_method in variation_neighborhoods
    do
        for coarsening_ratio in 0.0 0.1 0.3 0.5 0.7 1.0
        do
            c=$(echo $coarsening_ratio | sed 's/\.//g')
            python graph_data_analysis.py --dataset $dataset --coarsening_ratio $coarsening_ratio --coarsening_method $coarsening_method  --extra_node True
        done
    done
done