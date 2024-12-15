for i in {1..150}
do
    export i
    sbatch scripts/search.sh
done