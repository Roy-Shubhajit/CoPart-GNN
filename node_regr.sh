underscore="_"
extra="extra"
cluster="cluster"
coarsening_method="variation_neighborhoods"
for dataset in squirrel
do
    for coarsening_ratio in 0.9
    do 
        for exp_setup in Gs_train_2_Gs_infer
        do 
            c=$(echo $coarsening_ratio | sed 's/\.//g')
            output_dir=$dataset$underscore$c$underscore$coarsening_method$underscore$cluster
            python node_regression.py --dataset $dataset --coarsening_ratio $coarsening_ratio --coarsening_method $coarsening_method --exp_setup $exp_setup --cluster_node True --output_dir $output_dir
            output_dir=$dataset$underscore$c$underscore$coarsening_method$underscore$extra
            python node_regression.py --dataset $dataset --coarsening_ratio $coarsening_ratio --coarsening_method $coarsening_method --exp_setup $exp_setup --extra_node True --output_dir $output_dir
        done
    done
done