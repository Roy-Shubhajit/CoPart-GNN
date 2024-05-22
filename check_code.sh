underscore="_"
exp_fixed="fixed"
exp_few="few"
extra="extra"
dataset="cora"
coarsening_method=variation_neighborhoods
for coarsening_ratio in 0.1 0.3 0.5 0.7 1.0
do
    for exp_setup in Gc_train_2_Gs_train Gc_train_2_Gs_infer Gs_train_2_Gs_infer
    do 
        c=$(echo $coarsening_ratio | sed 's/\.//g')
        output_dir=$dataset$underscore$exp_fixed$underscore$exp_setup$underscore$c$underscore$coarsening_method$underscore$extra
        python main.py --dataset $dataset --experiment fixed --coarsening_ratio $coarsening_ratio --coarsening_method $coarsening_method --exp_setup $exp_setup --extra_node True --output_dir $output_dir
        output_dir=$dataset$underscore$exp_few$underscore$exp_setup$underscore$c$underscore$coarsening_method$underscore$extra
        python main.py --dataset $dataset --experiment few --coarsening_ratio $coarsening_ratio --coarsening_method $coarsening_method --exp_setup $exp_setup --extra_node True --output_dir $output_dir
    done
done