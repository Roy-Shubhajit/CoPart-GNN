#run train.py for different datasets and coarsening ratios
underscore="_"
exp_fixed="fixed"
exp_few="few"
exp_random="random"
extra="extra"
cluster="cluster"
transfer="transfer"
dataset="citeseer"
for coarsening_method in variation_neighborhoods algebraic_JC
do
    for coarsening_ratio in 0.0 0.1 0.3 0.5 0.7 1.0
    do
        c=$(echo $coarsening_ratio | sed 's/\.//g')
        output_dir=$dataset$underscore$exp_fixed$underscore$c$underscore$transfer$underscore$coarsening_method$underscore$cluster
        python train.py --dataset $dataset --experiment fixed --coarsening_ratio $coarsening_ratio --coarsening_method $coarsening_method  --cluster_node True --output_dir $output_dir 
        output_dir=$dataset$underscore$exp_few$underscore$c$underscore$transfer$underscore$coarsening_method$underscore$cluster
        python train.py --dataset $dataset --experiment few --coarsening_ratio $coarsening_ratio --coarsening_method $coarsening_method --cluster_node True --output_dir $output_dir
    done
done


