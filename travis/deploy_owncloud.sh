set -e -x

cd wheelhouse
for whl in essentia*.whl; do
    curl -X  PUT -u $DEPLOY_USER:"$DEPLOY_PASSWORD" "https://owncloud.rp.upf.edu/remote.php/webdav/$1/"$whl --data-binary @"$whl"    
done
cd ../dist
for sdist in essentia*.tar.gz; do
    curl -X  PUT -u $DEPLOY_USER:"$DEPLOY_PASSWORD" "https://owncloud.rp.upf.edu/remote.php/webdav/$1/"$sdist --data-binary @"$sdist"    
done
