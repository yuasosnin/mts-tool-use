python run_bench.py --model_id "Qwen/Qwen2-0.5B-Instruct" --bench_subset bfcl_simple --batch_size 32
python run_bench.py --model_id "Qwen/Qwen2.5-0.5B-Instruct" --bench_subset bfcl_simple --batch_size 32

python run_bench.py --model_id "Qwen/Qwen2-1.5B-Instruct" --bench_subset bfcl_simple --batch_size 4
python run_bench.py --model_id "Qwen/Qwen2.5-1.5B-Instruct" --bench_subset bfcl_simple --batch_size 4

python run_bench.py --model_id "Qwen/Qwen2-7B-Instruct" --bench_subset bfcl_simple --batch_size 4
python run_bench.py --model_id "Qwen/Qwen2.5-7B-Instruct" --bench_subset bfcl_simple --batch_size 4

python run_bench.py --model_id "Qwen/Qwen2-0.5B-Instruct" --bench_subset bfcl_multiple --batch_size 32
python run_bench.py --model_id "Qwen/Qwen2.5-0.5B-Instruct" --bench_subset bfcl_multiple --batch_size 32

python run_bench.py --model_id "Qwen/Qwen2-1.5B-Instruct" --bench_subset bfcl_multiple --batch_size 4
python run_bench.py --model_id "Qwen/Qwen2.5-1.5B-Instruct" --bench_subset bfcl_multiple --batch_size 4

python run_bench.py --model_id "Qwen/Qwen2-7B-Instruct" --bench_subset bfcl_multiple --batch_size 4
python run_bench.py --model_id "Qwen/Qwen2.5-7B-Instruct" --bench_subset bfcl_multiple --batch_size 4

python run_bench.py --model_id "Qwen/Qwen2-0.5B-Instruct" --bench_subset booking_multiple --batch_size 16
python run_bench.py --model_id "Qwen/Qwen2.5-0.5B-Instruct" --bench_subset booking_multiple --batch_size 16

python run_bench.py --model_id "Qwen/Qwen2-1.5B-Instruct" --bench_subset booking_multiple --batch_size 4
python run_bench.py --model_id "Qwen/Qwen2.5-1.5B-Instruct" --bench_subset booking_multiple --batch_size 4

python run_bench.py --model_id "Qwen/Qwen2-7B-Instruct" --bench_subset booking_multiple --batch_size 4
python run_bench.py --model_id "Qwen/Qwen2.5-7B-Instruct" --bench_subset booking_multiple --batch_size 4

python run_bench.py --model_id "yuasosnin/Qwen2-0.5B-Tool" --bench_subset bfcl_simple --batch_size 32
python run_bench.py --model_id "yuasosnin/Qwen2-0.5B-Tool" --bench_subset bfcl_multiple --batch_size 32
python run_bench.py --model_id "yuasosnin/Qwen2-0.5B-Tool" --bench_subset booking_multiple --batch_size 16
