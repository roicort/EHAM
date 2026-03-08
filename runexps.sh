#!/bin/bash

left_ds=mnist
right_ds=emnist
runpath=runs

echo "EAM Hetero experiments."
echo "Storing results in $runpath"
echo "=================== Starting at `date`"
# python eam.py -n $left_ds --runpath=$runpath && \    # -n: create_and_train_network (mnist)
# python eam.py -n $right_ds --runpath=$runpath && \   # -n: create_and_train_network (emnist)
# python eam.py -f $left_ds --runpath=$runpath && \      # -f: produce_features_from_data (mnist)
# python eam.py -f $right_ds --runpath=$runpath && \     # -f: produce_features_from_data (emnist)
python eam.py -c $left_ds --runpath=$runpath && \      # -c: characterize_features (mnist)
python eam.py -c $right_ds --runpath=$runpath && \     # -c: characterize_features (emnist)
python eam.py -s $left_ds --runpath=$runpath && \      # -s: run_separate_evaluation (mnist)
python eam.py -s $right_ds --runpath=$runpath && \     # -s: run_separate_evaluation (emnist)
python eam.py -e --runpath=$runpath && \               # -e: run_evaluation (test_hetero_fills)
# python eam.py -v --runpath=$runpath && \               # -v: no existe en eam.py actual
# python eam.py -w --runpath=$runpath && \               # -w: no existe en eam.py actual
python eam.py -q --runpath=$runpath && \               # -q: generate_memories (recall_with_cue)
python eam.py -r --runpath=$runpath && \               # -r: generate_memories (sampling + search)
python eam.py -P constructed --runpath=$runpath && \   # -P: recall_with_correct_proto (constructed)
python eam.py -P recalled --runpath=$runpath && \      # -P: recall_with_correct_proto (recalled)
python eam.py -P extracted --runpath=$runpath && \     # -P: recall_with_correct_proto (extracted)
python eam.py -p constructed --runpath=$runpath && \   # -p: recall_with_protos (constructed)
python eam.py -p recalled --runpath=$runpath && \      # -p: recall_with_protos (recalled)
python eam.py -p extracted --runpath=$runpath && \     # -p: recall_with_protos (extracted)
python eam.py -u --runpath=$runpath && \               # -u: generate_sequences
echo "=================== Ending at `date`"
ok=$?
if [ $ok -eq 0 ]; then
    echo "Done."
else
    echo "Sorry, something went wrong."
fi

