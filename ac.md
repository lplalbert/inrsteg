Dear Program Chairs, Senior Area Chairs and Area Chairs:

Thank you for your patient and thorough review of our submission! We are grateful for the reviewers' detailed and constructive feedback.

Due to the extensive nature of our responses, we prepare this message to **consolidate the reviewer comments and our rebuttals** for your review and decision-making process.



# Overview
The overall scores for our paper ranged from "borderline" to "accept", indicating a certain degree of controversy. In response, we submit a detailed rebuttal and provide a large amount of supplementary experimental data to address the reviewers' all questions. 

# Core Contribution

we proposes a pioneering approach by employing INRs to address the challenge of watermarking ultra-high resolution (UHR) images. 

We have summarised our strengths:
`Unmatched Efficiency`: We are the first watermarking method that can process UHR images with constant memory usage. Traditional CNN methods crash due to GPU memory overflow on large images, while our approach fundamentally solves the UHR computing bottleneck.

`Extreme Robustness`: Our core competitiveness lies in our ability to resist extreme cropping and scaling attacks. Even if only less than 1% of the image remains, we can still successfully decode the watermark information with an accuracy rate of over 98%. In the most rigorous tests, such as social media dissemination and mixed noise attacks, existing SOTA methods are almost ineffective, while our method remains robust, proving its reliability in complex real-world environments.

`Large-Scale Applications`: Our watermarks can be pre-generated offline and reused on a massive scale, making our method ideal for large image libraries, news agencies, content distribution platforms, and other scenarios that require the rapid addition of copyright information to millions of images.


Reviewers generally recognised the `novelty`  and `significance` of our paper.


# Controversy
The core controversy mainly revolves around the following `three aspects`.

## Slightly Lower Visual Quality
Due to the characteristics of INRs, our method is essentially a template-based watermark that does not utilise image content information, resulting in slightly lower visual quality in terms of PSNR/SSIM metrics compared to some CNN-based methods. Reviewer **2eV2, RwW9** expressed concern about this.


We first `emphasise` that our core contribution lies in proposing a novel INR-based paradigm that aims to fundamentally solve the scalability problem of CNN-based methods on UHR images, rather than optimising visual quality within the existing framework.

Secondly, we have thoroughly explored the hybrid model and adaptive embedding strength suggested by the reviewers, but experiments have proven that these alternatives are `not feasible.

We believe that, under the current INRs framework, sacrificing slight visual quality is an inherent and necessary `trade-off` in order to achieve `extreme robustness and scalability` under `limited computing resources`.

In addition, we responded to the advantages of template-based watermarks in large-scale image databases.


## Computational Efficiency & Cost
Reviewer **Ggtd, jFV8** had some questions about the computational cost of our method. This is mainly due to the inherently weaker representational ability of INRs compared to CNNs, which led the reviewer to `mistakenly` believe that our training cost is high.

To address this, we provide `explicit` training/inference times and GPU memory usage to demonstrate our computational cost advantages.

Overall, our training time is slightly longer than some lightweight CNN-based methods, but we outperform other methods in terms of `GPU memory cost`, `inference speed`, and other metrics.


## Comprehensiveness of Robustness Evaluation
Reviewer **2eV2, RwW9** mentions that the experiments lack an assessment of robustness against mixed distortion, real-world scenarios, and adversarial sample attacks.

Since our experimental design mainly follows previous `mainstream methods`, we do not focus on this type of attack. For this reason, we submit a large number of experiments to fill these gaps, and the results strongly support the advantages of our method.

Under the most stringent combination of `"noise+scaling+cropping"` attacks, the accuracy of other methods almost collapsed (~50%), while our method still maintained excellent performance.

In tests on platforms such as `Twitter and Instagram`, we achieve an accuracy rate of **92% to 99%**, far exceeding other methods. Even without design-specific conditions, we achieve an average accuracy rate of **91.2%** in printing scenarios.

Under attacks from `adversarial sample` methods such as PGD and FGSM, our method's decoding accuracy and image fidelity are significantly better than other SOTA methods.


# Conclusion
Based on this review, we believe that the main technical weaknesses and limitations raised by the reviewers have been effectively addressed.