
## Building

Vulkan validation layers can be installed from `https://vulkan.lunarg.com/sdk/home`

`git clone --recursive https://github.com/Reefufui/vulkan_image_processing.git`

`sh run.sh`


# ✔️Тривиальная проверка:

Достаточно создать папку `res` и поместить туда изображения, как указанно ниже и запустить `run.sh`

Запустить программу `./build/vulkan_denoice *имя папки с изображением*`

# ОS:

Works fine on my Arch Linux machine

# ✔️Реализация на CPU и GPU

Биальтеральный реализован на CPU и GPU

Нелокальный реализован только на GPU

# ✔️jИнформация в коммандной строке:

Время копирования и выполения шейдера (ns) для реализаций на GPU (Timestamps)

Время работы в секундах для CPU

# ✔️Реализованы два фильтра

Нелокальный и биальтеральный

# ✖️Учет допольнительных слоев

: ( не успел

# ✔️Учет соседних кадров

Это было смысл делать для нелокального фильтра - ведь чем больше похожих областей, тем лучше качество

Кол-во используемых кадров регулируется в исходном коде

# ✔️Перекрытие копирования и вычислений

Пока мы работаем с одним кадром - следующий уже копируется (в одном коммандном буффере)

Реализованно сменой DSetов - меням две текстуры (для копирования и для диспатча)

# ✔️Использование текстур

Имлеметирован режим для биальтерального фильтра в котором мы используем линейный буффер текселей, чтобы оценить разницу

# ✖️HDR изображения

: ( не успел

# Note
all images should be stored like this:

```
res
    image_name1
        frame-0.bmp
        frame-1.bmp
        frame-2.bmp
        frame-3.bmp
        frame-4.bmp
        frame-5.bmp
        frame-6.bmp
        frame-7.bmp
        frame-8.bmp
        frame-9.bmp
        layer-0.bmp
        layer-1.bmp
        layer-2.bmp
        layer-3.bmp
        layer-4.bmp
        layer-5.bmp
        layer-6.bmp
        layer-7.bmp
        layer-8.bmp
    image_name2
        frame-0.bmp
        frame-1.bmp
        frame-2.bmp
        frame-3.bmp
        frame-4.bmp
        frame-5.bmp
        frame-6.bmp
        frame-7.bmp
        frame-8.bmp
        frame-9.bmp
        layer-0.bmp
        layer-1.bmp
        layer-2.bmp
        layer-3.bmp
        layer-4.bmp
        layer-5.bmp
        layer-6.bmp
        layer-7.bmp
        layer-8.bmp
```

