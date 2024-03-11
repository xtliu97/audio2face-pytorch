# audio2face-pytorch

Pytorch implementation of audio generating face mesh or blendshape models. This repo now suppors models below:
- Audio2Mesh: Audio-Driven Facial Animation by Joint End-to-End Learning of Pose and Emotion. 
- VOCA: Capture, learning, and synthesis of 3D speaking styles.
- FaceFormer: Speech-Driven 3D Facial Animation with Transformers.

# Dataset
This repo use VOCASET as the template. 
'Capture, Learning, and Synthesis of 3D Speaking Styles' (CVPR 2019)
Also, `FLAME_sample` was extracted and converted to `asssets/FLAME_sample.obj` and Renderer is redesigned in. So that `psbody` lib is not nessary in this repo, which Apple Slicon users may have trouble when installing it.

# License
https://voca.is.tue.mpg.de/license.html


# References
- [VOCASET](https://voca.is.tue.mpg.de/license.html)
- Cudeiro, Daniel, et al. "Capture, learning, and synthesis of 3D speaking styles." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2019.
- [TimoBolkart/voca](https://github.com/TimoBolkart/voca)
- Fan, Yingruo, et al. "FaceFormer: Speech-Driven 3D Facial Animation with Transformers." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). 2022.
- NVIDIA. Audio-Driven Facial Animation by Joint End-to-End Learning of Pose and Emotion. 