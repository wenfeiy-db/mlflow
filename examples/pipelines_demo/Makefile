split_objects = split_train.parquet split_test.parquet split_summary.html

split: $(split_objects)

%_train.parquet %_test.parquet %_summary.html: datasets/autos.parquet
	mlx split-step \
		--input-path datasets/autos.parquet \
	  --summary-path $*_summary.html \
		--train-path $*_train.parquet \
		--test-path $*_test.parquet

transform_objects = transform_transformer.pkl transform_train_transformed.parquet

transform: $(transform_objects)

%_transformer.pkl %_train_transformed.parquet: steps/transform.py steps/transformer_config.yaml split_train.parquet
	mlx transform-step \
		--input-path split_train.parquet \
		--transformer-config steps/transformer_config.yaml \
		--transformer-path $*_transformer.pkl \
		--transformed-path $*_train_transformed.parquet

train_objects = train_pipeline.pkl train_run_id

train: $(train_objects)

%_pipeline.pkl %_run_id: steps/train.py steps/train_config.yaml transform_train_transformed.parquet transform_transformer.pkl
	mlx train-step \
		--input-path transform_train_transformed.parquet \
		--train-config steps/train_config.yaml \
		--transformer-path transform_transformer.pkl \
		--tracking-uri file:/tmp/mlruns \
		--pipeline-path $*_pipeline.pkl \
		--run-path $*_run_id

evaluate_objects = evaluate_results.html

evaluate: $(evaluate_objects)

%_results.html: train_pipeline.pkl split_train.parquet train_run_id
	mlx evaluate-step \
		--pipeline-path train_pipeline.pkl \
		--tracking-uri file:/tmp/mlruns \
		--run-id train_run_id \
		--train-path split_train.parquet \
		--evaluate-path $*_results.html

run: $(split_objects) $(transform_objects) $(train_objects) $(evaluate_objects)

clean:
	rm -rf $(split_objects) $(transform_objects) $(train_objects) $(evaluate_objects)
