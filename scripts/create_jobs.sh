for i in $(eval echo {1..$1})
do
  cat job-template.yml | sed "s/\$ITEM/$i/" > ./hyperparam-jobs-specs/hyp-job-$i.yml
done
