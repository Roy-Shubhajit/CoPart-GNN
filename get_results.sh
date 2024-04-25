#run train.py for different datasets and coarsening ratios
underscore="_"
exp_fixed="fixed"
exp_few="few"
exp_random="random"
extra="extra"
transfer="transfer"
for dataset in cora citeseer pubmed
do
    for coarsening_ratio in 0.1 0.3 0.5 0.7 0.9
    do
        c=$(echo $coarsening_ratio | sed 's/\.//g')
        output_dir=$dataset$underscore$exp_fixed$underscore$c$underscore$transfer$underscore$extra
        python train.py --dataset $dataset --experiment fixed --coarsening_ratio $coarsening_ratio --extra_node True --output_dir $output_dir
        output_dir=$dataset$underscore$exp_few$underscore$c$underscore$transfer$underscore$extra
        python train.py --dataset $dataset --experiment few --coarsening_ratio $coarsening_ratio --extra_node True --output_dir $output_dir
    done
done

for dataset in Physics dblp
do
    for coarsening_ratio in 0.1 0.3 0.5 0.7 0.9
    do
        c=$(echo $coarsening_ratio | sed 's/\.//g')
        output_dir=$dataset$underscore$exp_random$underscore$c$underscore$transfer$underscore$extra
        python train.py --dataset $dataset --experiment random --coarsening_ratio $coarsening_ratio --extra_node True --output_dir $output_dir
        output_dir=$dataset$underscore$exp_few$underscore$c$underscore$transfer$underscore$extra
        python train.py --dataset $dataset --experiment few --coarsening_ratio $coarsening_ratio --extra_node True --output_dir $output_dir
    done
done

