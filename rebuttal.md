# For Reviewer 2eV2

Thank you for your patience and thorough review! We note your concerns regarding visual quality.

It is important to emphasise that our `contribution` primarily lies in proposing a novel paradigm centred on INRs to address the challenge of robust embedding on ultra-large images.

We adopt this architecture specifically to overcome the computational bottlenecks and memory constraints that CNN-based methods face with high-resolution inputs. The slight reduction in visual fidelity stems from the inherent nature of INRs, which process pixel coordinates independently and thus do not leverage local spatial features as effectively as CNNs.

In fact, we made considerable efforts in this regard but encountered some challenges. We will detail these in the following sections.

## Q1: Using a hybrid INR-CNN model to improve perceptual quality

We appreciate the reviewer's insightful suggestion. Indeed, we explored similar hybrid architectures during our initial investigations but found them unsuitable due to several fundamental challenges.

we have explored hybrid architectures, using methods similar to LIIF [1], which involve first extracting features from the entire image using a lightweight CNN(3-layer Conv-BN-ReLU structure with 3×3 kernels), and then sampling these feature maps by coordinate for prediction with an INR.

It can be expressed as:

$$
L_c = \mathrm{CNN}(I)
$$

$$
z_e=\Gamma(L_c(x,y)\odot z_p)
$$

$L_c$ is similar to the embedded features mentioned in the section "Hierarchical Multi-Scale Coordinates Embedding" in the paper, while $z_e$ is equivalent to formula (4) in the paper.

However, this introduces two `issues`: (1) It is still necessary to perform CNN on the entire image, which results in computational overhead increasing with image size, and (2) computational efficiency is extremely low. In the CNN part, we need to perform calculations on the entire image, but in the INR part, we only sample local regions for watermark embedding during training, resulting in most of the computational overhead being meaningless.

If we only perform CNN on the specific regions where watermarks are to be embedded, This reintroduces the block-based generation problem shown in Figure 1 at our paper, requiring precise watermark block localization.

More importantly, since the final watermark generation still relies on the INR, during actual training, the network `selectively ignores` features from the CNN, degenerating into pure INR training.

we give our results on DIV2K(2K resolution):

|  Methods  | PSNR |  SSIM  | Avg ACC | GPU Memory |
| :-------: | :---: | :----: | :-----: | :--------: |
| Proposed | 37.27 | 0.9611 |  95.17  |   0.51GB   |
| LIIF-Like | 38.14 | 0.9652 |  94.64  |  12.34GB  |

Avg ACC represents the average decoding accuracy after various noise attacks, and GPU Memory represents the computing memory occupied by one sample during training. It can be seen that the visual quality has not improved significantly, while the GPU usage during training is much higher than that of our method.

In summary, we believe that this type of architecture may require further exploration.

## Q2: About adaptive $\gamma$

We have tried two strategies:

(1) Assigning a dynamic trainable parameter to each pixel, i.e., INR predicts not only RGB but also $\gamma$. However, this approach has issues where the predicted gamma values become completely ineffective, and due to the lack of hard constraints, this method produces very low visual quality.

(2) Pre-computing the texture richness of each pixel using the Laplacian operator and normalising it to [0,1] as $\gamma$. This approach effectively controls embedding only in high-frequency regions, but the decoding accuracy is very low. This is because INR cannot perceive the Laplacian operator, inevitably leading to information loss in low-frequency regions.

This is results:

|      Methods      | PSNR |  SSIM  | Avg ACC |
| :----------------: | :---: | :----: | :-----: |
|      Proposed      | 37.27 | 0.9611 |  95.17  |
| Predict$\gamma$ | 22.53 | 0.7298 |  97.33  |
| Laplacian Operator | 40.16 | 0.9811 |  78.41  |

## Q3: Adversarial examples or model inversion attacks

Currently, existing image watermarking methods do not extensively discuss this type of attack, so we primarily adopt mainstream attacks like DWSF[2] (ACM MM 23) and $\mathrm{RAIM_{ARK}}$[3] (ACM MM 24) in our design.

To our knowledge, "model inversion attacks" are mainly used to infer the original data used to train the model. This does not seem to be particularly relevant to the domain we are discussing.

In addition, we test the performance of different adversarial attack methods using the DIV2K dataset, where watermarks are embedded at 2K resolution with a fixed PSNR of 35 dB.

| Attack Method |   Proposed   |      |     DWSF     |      |   TrustMark   |      | $\mathrm{RAIM_{ARK}}$ |      |
| :-----------: | :-----------: | :---: | :-----------: | :---: | :-----------: | :---: | :---------------------: | :---: |
|              | Attacked PSNR |  ACC  | Attacked PSNR |  ACC  | Attacked PSNR |  ACC  |      Attacked PSNR      |  ACC  |
|     PGD1     |     34.90     | 99.99 |     32.88     | 25.17 |     38.20     | 95.41 |          30.61          | 76.94 |
|     PGD2     |     28.95     | 96.33 |     27.46     | 20.17 |     32.22     | 94.67 |          26.43          | 75.53 |
|     FGSM1     |     42.31     | 99.99 |     24.13     | 40.17 |     38.92     | 85.83 |          24.23          | 62.33 |
|     FGSM2     |     25.94     | 99.99 |      7.6      | 41.83 |     22.40     | 72.83 |          7.74          | 48.12 |
|     BIM1     |     36.92     | 99.99 |     18.75     | 13.50 |     33.04     | 88.17 |          25.12          | 65.73 |
|     BIM2     |     31.44     | 99.99 |     13.39     | 13.17 |     27.67     | 84.32 |          16.16          | 51.99 |

We used **torchattacks** to test three different attack methods: PGD[4], FGSM[5], and BIM[6]. Attacked PSNR represents the PSNR of the distorted image and the clean image after the attack,  ACC is the accuracy rate after the attack.

The specific settings are as follows:

```python
    pgd1 = torchattacks.PGD(wrapped_model, eps=8/255, alpha=2/255, steps=20)
    pgd2 = torchattacks.PGD(wrapped_model, eps=16/255, alpha=4/255, steps=20)
    fgsm1 = torchattacks.FGSM(wrapped_model, eps=16/255)
    fgsm2 = torchattacks.FGSM(wrapped_model, eps=128/255)
    bim1 = torchattacks.BIM(wrapped_model, eps=64/255, alpha=8/255, steps=20)
    bim2 = torchattacks.BIM(wrapped_model, eps=128/255, alpha=16/255, steps=20)
```

wrapped_model is the watermark decoder.

We find that our method demonstrates greater resistance to such attacks compared to other methods, as evidenced by two key factors: (1) Under identical attack parameters, our method maintains a higher PSNR after the attack, indicating its insensitivity to adversarial perturbations; (2) At the same PSNR level, our method achieves a higher accuracy rate, demonstrating its superior robustness against such attacks.

## Q4: Robustness of online platforms

We selected 50 images and tested our method on different media at PSNR 35 （2K resolution in DIV2K）:

|          |     Propsed     | DWSF | $\mathrm{RAIM_{ARK}}$ | TrustMark |
| :-------: | :-------------: | :---: | :---------------------: | :-------: |
|  Twitter  | **92.45** | 60.67 |          58.67          |   75.27   |
| Instagram | **99.67** | 76.33 |          66.13          |   75.4   |
|  Wechat  | **99.33** | 70.53 |          76.33          |   75.87   |

Our method is the most effective.

## Q5: Performance under extreme conditions

We test a set of extreme cases involving `"noise + scaling + cropping"`.

We first embed the messages into 2K images. These images are then subjected to various distortions. After that, we enlarge the distorted images to 4K resolution. Finally, we randomly crop 128×128 blocks from the enlarged images for decoding.

|                        |    Identity    |       GN       |       GF       |      JPEG      |     Dropout     |    Rotation    |   Translation   |      Color      |
| :---------------------: | :-------------: | :-------------: | :-------------: | :-------------: | :-------------: | :-------------: | :-------------: | :-------------: |
|         propsed         | **85.95** | **87.97** | **58.07** | **52.10** | **87.40** | **77.87** | **74.97** | **76.70** |
|          DWSF          |      50.02      |      49.9      |      50.73      |      50.16      |      49.16      |      50.73      |      50.12      |      50.16      |
|        TrustMark        |      53.99      |      51.34      |      51.25      |      50.75      |      50.25      |      50.75      |      50.54      |      48.99      |
| $\mathrm{RAIM_{ARK}}$ |      54.14      |      54.25      |      53.24      |      50.25      |      54.34      |      54.05      |      53.12      |      52.61      |

In this case, our performance will decline significantly, but it will still be far better than other solutions.

## Reference

[1] Chen Y, Liu S, Wang X. Learning continuous image representation with local implicit image function[C]//Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 2021: 8628-8638.

[2] H. Guo, Q. Zhang, J. Luo, F. Guo, W. Zhang, X. Su, and M. Li, “Practical deep dispersed watermarking with synchronization and fusion,” in Proceedings of the 31st ACM International Conference on Multimedia, 2023, pp. 7922–7932.

[3] Y. Wang, X. Zhu, G. Ye, S. Zhang, and X. Wei, “Achieving resolution-agnostic dnn-based image watermarking: A novel perspective of implicit neural representation,” in Proceedings of the 32nd ACM International Conference on Multimedia, 2024, pp. 10 354–10 362.

[4] Madry A, Makelov A, Schmidt L, et al. Towards deep learning models resistant to adversarial attacks[J]. arXiv preprint arXiv:1706.06083, 2017.

[5] Goodfellow I J, Shlens J, Szegedy C. Explaining and harnessing adversarial examples[J]. arXiv preprint arXiv:1412.6572, 2014.

[6] Kurakin A, Goodfellow I, Bengio S. Adversarial machine learning at scale[J]. arXiv preprint arXiv:1611.01236, 2016.

# For Reviewer RwW9

Thank you for your appreciation of our approach! Our main `contribution` lies in proposing a completely new paradigm to break through the inherent resolution problem. We apologise for any confusion caused by our oversight of certain details, and we will address each of your questions below.

## Q1: Discussion about Limited visual quality due to template-based watermarking approach

We have provided many **examples** of embeddings at different sizes in Figure 3 and Figure 7. As shown in Figure 3 (first row) shows that smaller image sizes with simpler textures exhibit more obvious visual distortion.

At the same time, we also explored how to utilise cover image content under the INR architecture. Our experimental methods are as follows:

(1) The RGB values of the original image and the feature$z_e$ in Section 3.1 are concatenated and fed into INR.

(2) Pre-computing the texture richness of each pixel using the Laplacian operator and normalising it to [0,1] as $\gamma$ (shown in Section 3.2). This approach effectively controls embedding only in high-frequency regions.

This is result at DIV2K:

| Methods | PSNR |  SSIM  | Avg ACC |
| :------: | :---: | :----: | :-----: |
| Proposed | 37.27 | 0.9611 |  95.17  |
|   (1)   | 37.31 | 0.9632 |  94.67  |
|   (2)   | 40.16 | 0.9811 |  80.41  |

As can be seen, The INR model selectively ignores raw RGB values, a reasonable approach given that isolated pixels do not convey meaningful semantic information on their own.

While using the Laplacian operator to constrain embedding to high-frequency regions improves visual quality, it significantly reduces decoding accuracy. This occurs because the Laplacian constraint ($\gamma$) and the predicted watermark residual are independent variables, making decoding accuracy unreliable.

We believe that introducing raw image features into the INR framework remains an area worthy of further exploration.

## Q2: Robustness of mixed distortions

For clarity, we have not included the performance of combination distortion in the main text.

Here is the `“noise+scaling+cropping"` results. The results represent the average accuracy rate of embedding on a 2K image(DIV2K datasets), first performing different distortion attacks, then scaling to different sizes, and finally randomly cropping a 128x128 patch for decoding. For fair comparison, PSNR is unified to 35.

|         128x128         |    Identity    |       GN       |       GF       |      JPEG      |     Dropout     |    Rotation    |   Translation   |      Color      |
| :---------------------: | :-------------: | :-------------: | :-------------: | :-------------: | :-------------: | :-------------: | :-------------: | :-------------: |
|         propsed         | **99.99** | **99.93** | **99.93** | **99.97** | **99.99** | **99.90** | **99.99** | **97.40** |
|          DWSF          |      90.17      |      89.76      |      89.4      |      87.36      |      76.4      |      85.56      |      65.9      |      78.36      |
|        TrustMark        |      87.65      |      83.72      |      82.6      |      75.22      |      75.7      |      49.5      |      62.13      |      87.25      |
| $\mathrm{RAIM_{ARK}}$ |      78.67      |      72.43      |      54.24      |      56.43      |      64.23      |      57.42      |      54.63      |      60.33      |

|         512x512         |    Identity    |       GN       |       GF       |      JPEG      |     Dropout     |    Rotation    |   Translation   |      Color      |
| :---------------------: | :-------------: | :-------------: | :-------------: | :-------------: | :-------------: | :-------------: | :-------------: | :-------------: |
|         propsed         | **99.86** | **99.89** | **99.57** | **98.90** | **99.85** | **99.81** | **99.81** | **98.07** |
|          DWSF          |      53.28      |      54.36      |      55.39      |      52.36      |      51.76      |      54.1      |      52.93      |      53.33      |
|        TrustMark        |      49.55      |      50.39      |      48.72      |      51.55      |      52.3      |      51.12      |      49.07      |      50.65      |
| $\mathrm{RAIM_{ARK}}$ |      53.24      |      53.1      |      52.53      |      52.54      |      54.21      |      50.53      |      49.2      |      50.54      |

|        2048x2048        |    Identity    |       GN       |       GF       |      JPEG      |     Dropout     |    Rotation    |   Translation   |      Color      |
| :---------------------: | :-------------: | :-------------: | :-------------: | :-------------: | :-------------: | :-------------: | :-------------: | :-------------: |
|         propsed         | **98.74** | **98.40** | **77.27** | **62.97** | **98.03** | **97.30** | **93.30** | **94.23** |
|          DWSF          |      50.83      |      50.93      |      50.7      |      50.5      |      49.93      |      50.92      |      50.21      |      50.87      |
|        TrustMark        |      52.5      |      51.43      |      51.25      |      50.75      |      50.25      |      50.75      |      50.5      |      51.25      |
| $\mathrm{RAIM_{ARK}}$ |      54.64      |      54.21      |      51.23      |      53.26      |      54.16      |      53.53      |      50.76      |      54.32      |

|        4096x4096        |    Identity    |       GN       |       GF       |      JPEG      |     Dropout     |    Rotation    |   Translation   |      Color      |
| :---------------------: | :-------------: | :-------------: | :-------------: | :-------------: | :-------------: | :-------------: | :-------------: | :-------------: |
|         propsed         | **85.95** | **87.97** | **58.07** | **52.10** | **87.40** | **77.87** | **74.97** | **76.70** |
|          DWSF          |      50.02      |      49.9      |      50.73      |      50.16      |      49.16      |      50.73      |      50.12      |      50.16      |
|        TrustMark        |      53.99      |      51.34      |      51.25      |      50.75      |      50.25      |      50.75      |      50.54      |      48.99      |
| $\mathrm{RAIM_{ARK}}$ |      54.14      |      54.25      |      53.24      |      50.25      |      54.34      |      54.05      |      53.12      |      52.61      |

Other methods cannot withstand resolutions higher than 512x512, while our method performs well under most combinations of noise.

## Q3: Discussion about stripe-like watermark pattern

We acknowledge this limitation may reduce concealment and make them vulnerable to targeted attacks and believe it represents a significant area for future improvement. During our early explorations, we found that using SIREN [1] as the base INR block or employing different parameter initialization methods for the Hierarchical Multi-Scale Coordinates Embedding resulted in `different watermark textures` (e.g., fractal-like effects). However, most of these approaches led to training failures where visual quality improved but decoding accuracy remained at 0.5.

Consequently, we chose the ReLU activation function and default parameter initialization as the most stable option. This finding suggests that model structure and initialization methods significantly influence watermark texture, making this a promising direction for future research.

## Q4: Discussion about variable-length messages

Our main goal is to effectively embed watermarks in high-resolution images using the INR paradigm, so this issue is somewhat beyond the scope of our discussion. We designed the message following `mainstream methods` such as HiDDeN[2] (2018), MBRS[3] (2021), DWSF[4] (2023), and $\mathrm{RAIM_{ARK}}$[5] (2024), all of which are fixed at 30 bits.

To the best of our knowledge, only training-free diffusion-based methods [6,7] can dynamically change embedding bit counts during inference. Other trained methods cannot modify bit counts at inference time.

Nevertheless, we can approximate dynamic bit changes through redundant embedding. For example, if trained with 6 bits but only 3 bits are needed (e.g., "010"), we embed "010|010" instead. This voting-based approach improves robustness. This method can improve robustness through voting.

## Q5: Format problem

Thank you for your careful review and attention to detail. We will fix these formatting issues in the camera-ready version.

## Reference

[1] Sitzmann V, Martel J, Bergman A, et al. Implicit neural representations with periodic activation functions[J]. Advances in neural information processing systems, 2020, 33: 7462-7473.

[2] Zhu J, Kaplan R, Johnson J, et al. Hidden: Hiding data with deep networks[C]//Proceedings of the European conference on computer vision (ECCV). 2018: 657-672.

[3] Jia Z, Fang H, Zhang W. Mbrs: Enhancing robustness of dnn-based watermarking by mini-batch of real and simulated jpeg compression[C]//Proceedings of the 29th ACM international conference on multimedia. 2021: 41-49.

[4] H. Guo, Q. Zhang, J. Luo, F. Guo, W. Zhang, X. Su, and M. Li, “Practical deep dispersed watermarking with synchronization and fusion,” in Proceedings of the 31st ACM International Conference on Multimedia, 2023, pp. 7922–7932.

[5] Y. Wang, X. Zhu, G. Ye, S. Zhang, and X. Wei, “Achieving resolution-agnostic dnn-based image watermarking: A novel perspective of implicit neural representation,” in Proceedings of the 32nd ACM International Conference on Multimedia, 2024, pp. 10 354–10 362.

[6] Wen Y, Kirchenbauer J, Geiping J, et al. Tree-ring watermarks: Fingerprints for diffusion images that are invisible and robust[J]. arXiv preprint arXiv:2305.20030, 2023.

[7] Yang Z, Zeng K, Chen K, et al. Gaussian shading: Provable performance-lossless image watermarking for diffusion models[C]//Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2024: 12162-12171.

# For Reviewer Ggtd

Thank you for your patience in reviewing our work! We have noticed that you seem to have some questions about the computational efficiency of our method, and we sincerely apologise for any confusion caused by unclear wording.

In the main text, we primarily aimed to highlight that INR has weaker representational capabilities compared to CNN. As a result, under the same scale, INR-based methods require longer training times and converge more slowly. However, since INR is essentially an MLP, its computational overhead during inference is significantly lower than that of CNN under the same scale. We will provide a detailed explanation in the following sections.

## Q1: Robustness Analysis Across Different Spatial Locations

Due to our coordinate sampling strategy, our method can guarantee `decoding at any position`.
To prove this, we embedded the watermark in 2K images (DIV2K dataset) and cropped blocks ranging from 128×128 to 1024×1024 pixels from five different positions (top left, bottom left, top right, bottom right, and center), then calculated the average accuracy rate of blocks(PSNR: 35 dB).

|   Position   | 128x128 | 512x512 | 1024x1024 |
| :----------: | :-----: | :-----: | :-------: |
|   top left   |  99.93  |  99.99  |   99.99   |
|  top right  |  99.83  |  99.99  |   99.99   |
| bottom left |  99.87  |  99.99  |   99.99   |
| bottom right |  99.83  |  99.99  |   99.99   |
|    center    |  99.73  |  99.99  |   99.99   |

As can be seen, our method can robustly decode regardless of the position.

## Q2: Clarification on INR Fitting Process and Computational Cost

We sincerely apologise for any confusion caused by our unclear statement.

When we say that "the fitting process of INR can be computationally expensive," we mean that the `model converges` slowly during training. This is because INR has weaker representation capabilities than CNN, and our method requires more time to train than CNN-based methods. However, once training is complete, our method can not only pre-generate watermark templates, but also process arbitrary pixels independently and in parallel, resulting in high computational efficiency.

Here is a comparison of training times using one RTX 3090 24G (32 batch):

| Method | Proposed | HiDDeN | StegaStamp | MBRS | DWSF | TrustMark | $\mathrm{RAIM_{ARK}}$ |
| :----: | :------: | :----: | :--------: | :--: | :--: | :-------: | :---------------------: |
| Times |   26h   |  23h  |    19h    | 14h | 39h |     -     |    >20min per image    |

Although our method is slower than methods such as MBRS, it is still faster than DWSF. Furthermore, $\mathrm{RAIM_{ARK}}$ requires fine-tuning for each image, which is unacceptable in large-scale production.

## Q3: Report the actual time required to generate a single watermark during embedding

Unlike CNNs, which require computations across the entire image, our method enables independent parallel processing for each pixel. Consequently, in theory, given sufficient GPU resources for parallel execution, our computational overhead is equivalent to the inference time of a single pixel.

We demonstrate this by measuring the inference time for watermarking a single 2K image using one GPU, two GPUs(RTX3090 24GB):

|      |    Ours(1 GPU)    |    Ours(2 GPU)    | DWSF | TrustMark | $\mathrm{RAIM_{ARK}}$ |
| :---: | :---------------: | :---------------: | :--: | :-------: | :---------------------: |
| Times | **24.16ms** | **13.74ms** | 74ms |   308ms   |    >20min per image    |

Our INR method requires only 24.16ms on a single GPU, which is 3x faster than DWSF and 12.7x faster than TrustMark. Essentially, INR is just MLP, and its computational load is much lower than that of CNN of the same scale.

# For Reviewer jFV8

Thank you for your appreciation of our work! We will answer your questions one by one.

## Q1: Capacity of Framework

The fixed 30-bit length follows the conventional paradigm established in previous works, which is widely adopted in both low-resolution studies such as HiDDeN[1] (2018) and MBRS[2] (2021), and high-resolution approaches like DWSF[3] (2023) and $\mathrm{RAIM_{ARK}}$[4] (2024).

We present the relationship between visual quality and decoding accuracy at 30, 50, and 100 bits:

| num of bit | PSNR |  SSIM  | Avg ACC |
| :--------: | :---: | :----: | :-----: |
|     30     | 37.27 | 0.9656 |  95.76  |
|     50     | 35.59 | 0.9563 |  95.62  |
|    100    | 31.32 | 0.9123 |  88.26  |

In addition, We find that convergence was quite difficult at 100 bits (the convergence rate is slow), and the visual quality deteriorated significantly.

This is reasonable. In order to accurately extract watermarks at any size, it is inevitable that a large amount of redundant information will be embedded. Therefore, how to further improve capacity based on existing methods remains a topic worthy of further exploration.

## Q2: Missing of local packages and instructions

We apologize, but since FastTools is an internal package that contains a large amount of private information (such as Hugging Face user tokens and passwords), we have not yet organised it. We will organise clean code and clear instructions after the camera-ready version is complete and open source it.

## Q3: Compare of training time

Here is a comparison of training times using one RTX 3090 24G (32 batch):

| Method | Proposed | HiDDeN | StegaStamp | MBRS | DWSF | TrustMark | $\mathrm{RAIM_{ARK}}$ |
| :----: | :------: | :----: | :--------: | :--: | :--: | :-------: | :---------------------: |
| Times |   26h   |  23h  |    19h    | 14h | 39h |     -     |    >20min per image    |

As can be seen, CNN-based methods have a clear advantage in terms of training time, but we are still significantly faster than DWSF in training. Furthermore, $\mathrm{RAIM_{ARK}}$ requires fine-tuning for each image, which is unacceptable in large-scale production.

## Q4: Hardware specification

We provide detailed specifications for hardware and software.

|        |                                          |
| :----: | :---------------------------------------: |
|  GPU  |        NVIDIA GeForce RTX 3090 24G        |
|  CPU  | Intel(R) Xeon(R) Platinum 8269CY @ 2.5GHz |
| Memery |  Samsung M393A4K40CB2-CTD DDR4 32G x 12  |
|   OS   |            Ubuntu 20.04.1 LTS            |
|  CUDA  |                   12.8                   |

## Q5: Printing images distortion adaptation

StegaStamp is able to cope with printing image distortion mainly because it implements noise layers that simulate this process. Therefore, we only need to replace our noise layer with theirs, which enables adaptation to printing image distortion.

Additionally, our designed distortion layers already encompass their design to a certain extent. Therefore, we directly present our experimental results.

we randomly select 50 2K images (PSNR 35), print them on A4 paper using an **HP Color Laser MFP 178nw**, and captured them using a **Redmi K20 Pro** at a distance of 20 cm from the image. The average accuracy rate we achieved was `91.2%`.

## Q6: Format problem

Thank you for your patience and attention to detail. We will correct these formatting errors in the camera-ready version.

## Reference

[1] Zhu J, Kaplan R, Johnson J, et al. Hidden: Hiding data with deep networks[C]//Proceedings of the European conference on computer vision (ECCV). 2018: 657-672.

[2] Jia Z, Fang H, Zhang W. Mbrs: Enhancing robustness of dnn-based watermarking by mini-batch of real and simulated jpeg compression[C]//Proceedings of the 29th ACM international conference on multimedia. 2021: 41-49.

[3] H. Guo, Q. Zhang, J. Luo, F. Guo, W. Zhang, X. Su, and M. Li, “Practical deep dispersed watermarking with synchronization and fusion,” in Proceedings of the 31st ACM International Conference on Multimedia, 2023, pp. 7922–7932.

[4] Y. Wang, X. Zhu, G. Ye, S. Zhang, and X. Wei, “Achieving resolution-agnostic dnn-based image watermarking: A novel perspective of implicit neural representation,” in Proceedings of the 32nd ACM International Conference on Multimedia, 2024, pp. 10 354–10 362.

# For  Reviewer v2Wa

Thank you for your high praise of our work! We apologize for any confusion that may have been caused by unclear wording in the main text. We will make the necessary corrections in the camera-ready version.

## Q1: Lack of Detailed Computational Complexity Analysis (in Weakness)

We have already provided the inference time in Table 4. We mainly claim that we can train images of any size with fixed and limited computing resources. To this end, we provide the GPU usage (GB) per sample during training (Adam optim) for different methods:

|          |    Proposed    | HiDDeN |   StegaStamp   | MBRS | DWSF | $\mathrm{RAIM_{ARK}}$ |
| :-------: | :------------: | :----: | :------------: | :--: | :--: | :---------------------: |
|  128x128  |      0.51      |  0.18  | **0.07** | 0.47 | 1.8 |          1.54          |
|  512x512  | **0.51** |  5.04  |      0.75      | 4.73 |  -  |          11.93          |
| 1024x1024 | **0.51** |  7.42  |     2.628     | 17.4 |  -  |          > 24          |
| 2048x2048 | **0.51** |  >24  |     10.51     | >24 |  -  |          > 24          |

It can be seen that the memory usage of our method does not increase with size.

## Q2: the selection of rank influence the balance between computational efficiency and watermark performance

In Section 4.6, we have already shown the impact of different ranks on visual quality and decoding accuracy in the ablation experiment.

Here, we focus on the GPU memory consumption of different ranks with the corresponding number of INR parameters in training single samples, and we also provide the GPU memory consumption without using low-rank watermarking for watermark injection:

|                    | w/o low-rank |  r=8  | r=32 | r=64 | r=128 | r=256 |
| :-----------------: | :----------: | :---: | :---: | :--: | :---: | :---: |
| GPU consumption(GB) |     > 24     | 0.49 | 0.51 | 0.53 | 0.60 | 0.71 |
|   INR parameters   |    34.6M    | 22.6k | 84.2k | 166k | 330k | 659k |

If we directly use formula (5) from the paper, single sample training would exceed 24GB memory. Since we use RTX 3090 with 24GB, this makes training impossible to complete.

## Q3: Interval Selection Process and Consistency Guarantee

For simplicity in the main text, we refer to this as a "randomly selected interval $\Delta_t$". However, in our actual implementation, $\Delta_t$ is automatically determined based on the selected region.

We first randomly sample the top-left coordinates $(x,y)$ from the "normalised coordinate matrix" (shown in section 3.1) and randomly sample the submatrix side length $s\in[0,2]$. Then, we generate the subcoordinate matrix using the fixed grid parameter $r$. The code is as follows:

```python
    x = torch.linspace(top_left[0], top_left[0] + s, r)
    y = torch.linspace(top_left[1], top_left[1] + s, r)
    xv, yv = torch.meshgrid(x, y, indexing='ij')
    coordinates = torch.stack([xv, yv], dim=-1)
```

As can be seen, as long as we sample a matrix of any size from the "normalised coordinate matrix", we can determine a corresponding $\Delta_t$.

Consistency is ensured by the intrinsic properties of INRs, which learn a continuous coordinate-to-RGB mapping. Since each coordinate maps to a unique RGB value and coordinates are processed independently during training, the same coordinate will always yield the same RGB value regardless of the sampling submatrix's size or location, thereby maintaining consistency across scales.

## Q4: Specific image dimensions used for the high-resolution and low-resolution categories in Table 1

For High Resolution, we measured at 2K size (except for DWSF), and for Low Resolution, we measured at 128x128 size. For DWSF, it essentially embeds the watermark into many 128x128 blocks, so we measured the visual quality of the block.

We understand your concern that visual quality may vary across different resolutions, so we have also measured the visual quality of our method at different resolutions:

|      | 128x128 | 512x52 | 1024x1024 | 2048x2048 |
| :--: | :-----: | :----: | :-------: | :-------: |
| PSNR |  37.46  | 37.56 |   37.34   |   37.27   |
| SSIM | 0.9621 | 0.9632 |  0.9613  |  0.9611  |

Our method maintains consistent visual quality at different resolutions.

## Q5: Whether a single pretrained model is applicable across all resolutions without retraining

As mentioned in Section 4.3, in order to make a fair comparison with the low-resolution model, Table 2 shows the results of training at a fixed resolution of 128×128 only.

Table 3 and Figure 4 shows the performance of our method at varying image resolutions. It is trained `only` at 2K resolution and tested at different resolutions.

We will clarify this point in the camera-ready version.
