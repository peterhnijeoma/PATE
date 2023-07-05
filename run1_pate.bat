echo "running for epsilon = 0.2"
echo
echo "running for 250 0.2 SVHN"
python pate_demo.py 250 0.2 true SVHN

echo "running for 250 0.2 FMNIST"
python pate_demo.py 250 0.2 true FMNIST

echo "running for 250 0.2 MNIST"
python pate_demo.py 250 0.2 true MNIST

echo "running with no noise added"
echo
echo "running for 250 0.0 false SVHN"
python pate_demo.py 250 0.0 false SVHN

echo "running for 250 0.0 false FMNIST"
python pate_demo.py 250 0.0 false FMNIST

echo "running for 250 0.0 false MNIST"
python pate_demo.py 250 0.0 false MNIST
