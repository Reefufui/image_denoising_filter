echo "compiling shaders..."
glslangValidator -V nonlocal.comp -o nonlocal.spv
glslangValidator -V normalize.comp -o normalize.spv
glslangValidator -V bialteral.comp -o bialteral.spv
glslangValidator -V bialteral_linear.comp -o bialteral_linear.spv
glslangValidator -V bialteral_layers.comp -o bialteral_layers.spv
