#ifndef BITMAP_GUARDIAN_H
#define BITMAP_GUARDIAN_H

struct Pixel
{
    unsigned char r, g, b;
};

/**
  \brief save 24 bit RGB bitmap images.
  \param fname  - file name
  \param w      - input image width
  \param h      - input image height
  \param pixels - R8G8B8A8 data, 4 bytes per pixel, one byte for channel.
  */
void SaveBMP(const char* fname, const unsigned int* pixels, int w, int h);
void WriteBMP(const char* fname, Pixel* a_pixelData, int width, int height);

/**
  \brief load 24 bit RGB bitmap images.
  \param fname - file name
  \param pW    - out image width
  \param pH    - out image height
  \return R8G8B8A8 data, 4 bytes per pixel, one byte for channel.

  Note that this function in this sample works correctly _ONLY_ for 24 bit RGB ".bmp" images.
  If you want to support gray-scale images or images with palette, please upgrade its implementation.
  */
std::vector<unsigned int> LoadBMP(const char* fname, int* pW, int* pH);
std::vector<Pixel> LoadBMPpix(const char* fname, int* pW, int* pH);

#endif
