set -e -x

cd wheelhouse
for whl in essentia-*.whl; do
    curl --upload-file "$whl"  https://transfer.sh/"$whl" -w "\n"
done



