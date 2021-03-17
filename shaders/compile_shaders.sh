echo "compiling shaders..."
glslangValidator -V nonlocal.comp -o nonlocal.spv
glslangValidator -V nonlocal_generate.comp -o nonlocal_generate.spv
glslangValidator -V bialteral.comp -o bialteral.spv
glslangValidator -V bialteral_linear.comp -o bialteral_linear.spv
