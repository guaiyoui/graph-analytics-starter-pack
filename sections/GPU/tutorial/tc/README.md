# Triangle counting on gpu sample code

## Compile
```bash
mkdir build
cd build
cmake ..
make
cd ..
```

## Run
```bash
./build/tc --graph data/ucidata-zachary/out.ucidata-zachary --gpu 0
```