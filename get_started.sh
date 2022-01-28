
wget https://www.robots.ox.ac.uk/~vadim/neuraldiff/release/ckpts.tar.gz

tar -xzvf ckpts.tar.gz

mkdir data

cd data

wget https://www.robots.ox.ac.uk/~vadim/neuraldiff/release/EPIC-Diff-annotations.tar.gz

tar -xzvf EPIC-Diff-annotations.tar.gz

wget https://data.bris.ac.uk/datasets/tar/296c4vv03j7lb2ejq3874ej3vm.zip

unzip 296c4vv03j7lb2ejq3874ej3vm.zip

export EKPATH=296c4vv03j7lb2ejq3874ej3vm

for X in $(ls $EKPATH);
  do echo $X;
  for Z in $(ls $EKPATH/$X);
    do echo $Z;
    # ln -s $PWD/$EKPATH/$X/$Z EPIC-Diff/$X
    mv $PWD/$EKPATH/$X/$Z EPIC-Diff/$X
    # unlink EPIC-Diff/$X/$Z
  done;
done

# unlink EPIC-Diff/readme.txt
# ln -s $PWD/$EKPATH/readme.txt EPIC-Diff/README_EPIC-Kitchens.txt

mv $PWD/$EKPATH/readme.txt EPIC-Diff/README_EPIC-Kitchens.txt
