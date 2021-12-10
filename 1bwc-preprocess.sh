#!/bin/bash
i=1
for filename in train_data/training-monolingual.tokenized.shuffled/*; do
      echo "file nb: $i, current file: $filename"
      cat $filename | python pre-process.py -v=1 --min=5 --max=25 \
                  --no='" - -- # www http' \
                  -o=train_data/preprocessed-full/$(basename $filename).ref \
                  --lower > train_data/preprocessed-full/$(basename $filename).test
      ((i+=1))
done

cat train_data/heldout/news.en-00000-of-00100 | python pre-process.py -v=1 --min=5 --max=25 \
                  --no='" - -- # www http' \
                  -o=train_data/heldout/news.en-00000-of-00100.ref \
                  --lower > train_data/heldout/news.en-00000-of-00100.test