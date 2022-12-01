---
title : "Logo Detection"
output:   
    md_document:
        variant: markdown_github
bibliography: "references.bib"
#nocite: '@*'
---

<!-- Add banner here -->

# W281 - Final Project : Logo Detection

<!-- ![](./all_logos.png  {width=40px height=400px} ) -->
<img src="./all_logos.png" width="400" height="200" >

Authors: Luis Chion (lmchion@berkeley.edu), Eric Liu (eliu390@berkeley.edu), Viswanathan Thiagarajan (viswanathan@berkeley.edu), John Calzaretta (john.calzaretta@berkeley.edu)

Instructor: Allen Y. Yang

Date: 11/23/2022


---
<!-- Add buttons here -->

<!-- Describe your project in brief -->

<!-- The project title should be self explanotory and try not to make it a mouthful. (Although exceptions exist- **awesome-readme-writing-guide-for-open-source-projects** - would have been a cool name)

Add a cover/banner image for your README. **Why?** Because it easily **grabs people's attention** and it **looks cool**(*duh!obviously!*).

The best dimensions for the banner is **1280x650px**. You could also use this for social preview of your repo.

I personally use [**Canva**](https://www.canva.com/) for creating the banner images. All the basic stuff is **free**(*you won't need the pro version in most cases*).

There are endless badges that you could use in your projects. And they do depend on the project. Some of the ones that I commonly use in every projects are given below. 

I use [**Shields IO**](https://shields.io/) for making badges. It is a simple and easy to use tool that you can use for almost all your badge cravings. -->

<!-- Some badges that you could use -->

<!-- ![GitHub release (latest by date including pre-releases)](https://img.shields.io/github/v/release/navendu-pottekkat/awesome-readme?include_prereleases)
: This badge shows the version of the current release.

![GitHub last commit](https://img.shields.io/github/last-commit/navendu-pottekkat/awesome-readme)
: I think it is self-explanatory. This gives people an idea about how the project is being maintained.

![GitHub issues](https://img.shields.io/github/issues-raw/navendu-pottekkat/awesome-readme)
: This is a dynamic badge from [**Shields IO**](https://shields.io/) that tracks issues in your project and gets updated automatically. It gives the user an idea about the issues and they can just click the badge to view the issues.

![GitHub pull requests](https://img.shields.io/github/issues-pr/navendu-pottekkat/awesome-readme)
: This is also a dynamic badge that tracks pull requests. This notifies the maintainers of the project when a new pull request comes.

![GitHub All Releases](https://img.shields.io/github/downloads/navendu-pottekkat/awesome-readme/total): If you are not like me and your project gets a lot of downloads(*I envy you*) then you should have a badge that shows the number of downloads! This lets others know how **Awesome** your project is and is worth contributing to.

![GitHub](https://img.shields.io/github/license/navendu-pottekkat/awesome-readme)
: This shows what kind of open-source license your project uses. This is good idea as it lets people know how they can use your project for themselves.

![Tweet](https://img.shields.io/twitter/url?style=flat-square&logo=twitter&url=https%3A%2F%2Fnavendu.me%2Fnsfw-filter%2Findex.html): This is not essential but it is a cool way to let others know about your project! Clicking this button automatically opens twitter and writes a tweet about your project and link to it. All the user has to do is to click tweet. Isn't that neat? -->

# Table of contents 

<!-- After you have introduced your project, it is a good idea to add a **Table of contents** or **TOC** as **cool** people say it. This would make it easier for people to navigate through your README and find exactly what they are looking for.

Here is a sample TOC(*wow! such cool!*) that is actually the TOC for this README. -->

- [W281 - Final Project : Logo Detection](#w281---final-project--logo-detection)


- [Table of contents](#table-of-contents)


- [Overview](#overview)


- [Related work](#related-work)


- [Dataset](#dataset)



- [Methods](#methods)


  - [Data Preprocessing](#data-preprocessing)

  
  
  - [General Feature Extraction](#general-feature-extraction)
  - [SIFT](#sift)
  - [YOLO](#yolo)
- [Results and Discussion](#results-and-discussion)
- [References](#references)

<!-- - [Usage](#usage)
<!-- - [Dataset](#dataset)
<!-- - [Contribute](#contribute)
    - [Sponsor](#sponsor)
    - [Adding new features or fixing bugs](#adding-new-features-or-fixing-bugs)
<!--
 - [License](#license) -->
<!-- - [Footer](#footer) --> 



# Overview


Brand logos can be found nowadays almost everywehere from images produced by IoT devices (cars and survellaince cameras) to social media postings (Facebook, Tiktok, Instagram).  As such, logo recognition is a fundamental problem for computer vision and can be used in the following applications:
- Copyright infringment : to detect similar logo patterns and colors as a well recognized brand
- Brand related statistics :  to understand consumer for targetted advertising
- Intelligent traffic-control systems :  to recognize a stop sign  using camera feed from vehicles

Logo recognition can be considered a subset of object recognition.  In general, the majority of the logos are two dimensional objects containing stylized shapes, no texture and primary colors.  In some cases, logos can contain text (i.e. Fedex logo) or can be placed in different surfaces (i.e. Coca-Cola bottle or Adidas shoes).  The logo detection process can be split in two tasks:  determining the location of the logo (bounding box) and logo classification.  Finding the logo in a real world image is a challenging task.  It is desirable to have a incremental logo model learning without exhaustive manual labelling of increasing data expansion [[5]](#5). Also, logos can appear in highly diverse contexts, scales, changes in illumination, size, resolution, and perspectives [[6]](#6)


<!-- Logos are persistent advertisements for a brand that may be mobile (printed on consumer goods) or immobile (storefront). In a streetview image of an area, visible logos potentially contain information about brand market share and the area's socioeconomic status. This idea can be combined with a rapidly growing resource: all kinds of devices that are equipped with cameras which constantly stream visual data to the cloud. Since this data can be automatically location-tagged, this is a rich data source for fraud detection, predicting consumer trends, or analyzing the socioeconomic status of an area based on logo type and frequency. This requires a computer vision algorithm that can identify unlabeled logos in a visual scene. -->



<!-- # Overview -->





<!-- Add a demo for your project -->

<!-- After you have written about your project, it is a good idea to have a demo/preview(**video/gif/screenshots** are good options) of your project so that people can know what to expect in your project. You could also add the demo in the previous section with the product description.

Here is a random GIF as a placeholder.

![Random GIF](https://media.giphy.com/media/ZVik7pBtu9dNS/giphy.gif) -->




# Related work
<!-- [(Back to top)](#table-of-contents) --> 


See [[1]](#1) and [[2]](#2) reference

<!-- *You might have noticed the **Back to top** button(if not, please notice, it's right there!). This is a good idea because it makes your README **easy to navigate.*** 

The first one should be how to install(how to generally use your project or set-up for editing in their machine).

This should give the users a concrete idea with instructions on how they can use your project repo with all the steps.

Following this steps, **they should be able to run this in their device.**

A method I use is after completing the README, I go through the instructions from scratch and check if it is working. -->

<!-- Here is a sample instruction:-->

@@@Discuss YOLO@@@


# Dataset

![Logos-32plus](http://www.ivl.disco.unimib.it/activities/logo-recognition/) is a collection of 12,312 real-world photos that contain 32 different logo classes. It is an expansion on the the FlickrLogos-32 dataset, designed to be more representative of the various conditions that logos appear in and more suitable for training keypoint-based approaches to logo recognition. To construct this dataset, images were scraped from Flickr and Google Images through a text-based search on image tags. Scraped images were manually filtered to remove unfocused, blurred, noisy, or duplicate images [1].

We selected 10 logo classes from the 32 in Logos-32plus to use as the total dataset for training and evaluation. Each image in this dataset is labeled with a single class, and the 10 classes each contain 300 photos on average. Bounding box annotations are provided for each occurrence of a logo in an image; photos may have one or multiple instances of the logo corresponding to the labeled class. In cases where an image contains logos belonging to multiple classes, only logos corresponding to the image class are annotated. The counts of images and bounding boxes per class are shown in Fig. 1. 

![alt text](https://github.com/jcalz23/logo_detection_w281/blob/main/images/class_counts.png?raw=true)
**Fig. 1.** Counts of images and individual logo bounding boxes for each class. Note that bounding box counts are significantly larger than image counts due to images containing multiple occurrences of a logo.

<!-- [(Back to top)](#table-of-contents)  --> 

<!-- This is optional and it is used to give the user info on how to use the project after installation. This could be added in the Installation section also. -->





# Methods
<!-- [(Back to top)](#table-of-contents) -->

<!-- This is the place where you give instructions to developers on how to modify the code.

You could give **instructions in depth** of **how the code works** and how everything is put together.

You could also give specific instructions to how they can setup their development environment.

Ideally, you should keep the README simple. If you need to add more complex explanations, use a wiki. Check out [this wiki](https://github.com/navendu-pottekkat/nsfw-filter/wiki) for inspiration. -->

Non-deep learning approaches to logo prediction typically involve two steps. The first step is logo localization, which identifies potential locations in an image where a logo may be present. This can be solved by using a correlation tracker to detect image regions that have high correlation with a mask image. Multiple mask images are produced from each logo via affine transformation, rotation, and resizing. Regions that produce a sufficiently high correlation with any mask image are annotated with a bounding box and the class of the mask image. The second step is logo classification, which uses a feature-based model to perform image recognition on the bounding boxes produced by the logo localization step. Since the Logos-32plus dataset provides ground truth bounding boxes for all images, this project assumes a strongly-labeled dataset as input and focuses only on the logo classification problem.

The YOLO deep learning model takes weakly-labeled images as input, while our classification system takes labeled bounding boxes as input and performs data augmentation. The train-validation-test split is applied at the image level before preprocessing to ensure that all training images produced by data augmentation are generated from the training images seen by YOLO; similarly for validation and test. We apply a 70-15-15 train-validation-test split in order to maximize the size of the validation and test sets and increase the generalization and robustness of model evaluation metrics. The impact of the reduced training set size is negligible due to the significant data augmentation performed.


## Image Preprocessing

Using the ground truth bounding boxes provided by the Logos-32plus dataset, all bounding boxes are extracted from each image. Each bounding box is now considered a unique logo image that belongs to the same class and data split as the source image. Image contrast normalization is performed on by applying the Contrast Limited Adaptive Histogram Equalization (CLAHE) algorithm with 4x4 tile size to the luminance channel of each image. Example output of this step is shown in Fig. 2.

![alt text](https://github.com/jcalz23/logo_detection_w281/blob/main/images/adidas_clahe.png?raw=true)
**Fig. 2.** 

Data augmentation generates additional training examples from a single image by applying random 3D rotation transformations and color inversions. Class balancing is enforced by adjusting the number of generated images such that the final image counts are uniform. The total number of images after data augmentation is abcdefg.


## General Feature Extraction 

## SIFT

## YOLO


# Results and Discussion


# References 


<a id="1">[1]</a> 
Simone Bianco, Marco Buzzelli, Davide Mazzini and Raimondo Schettini (2017).
Deep Learning for Logo Recognition. Neurocomputing.
https://doi.org/10.1016/j.neucom.2017.03.051


<a id="2">[2]</a> 
Choras, Ryszard S.. (2007). Image feature extraction techniques and their applications for CBIR and biometrics systems. 
International Journal of Biology and Biomedical Engineering.  
https://www.researchgate.net/publication/228711889_Image_feature_extraction_techniques_and_their_applications_for_CBIR_and_biometrics_systems

<a id="3">[3]</a> 
Stefan Romberg and Rainer Lienhart. 2013. Bundle min-hashing for logo recognition. In Proceedings of the 3rd ACM conference on International conference on multimedia retrieval (ICMR '13). Association for Computing Machinery, New York, NY, USA, 113–120. https://doi.org/10.1145/2461466.2461486

<a id="4">[4]</a> 
C. Li, I. Fehérvári, X. Zhao, I. Macedo and S. Appalaraju, "SeeTek: Very Large-Scale Open-set Logo Recognition with Text-Aware Metric Learning," 2022 IEEE/CVF Winter Conference on Applications of Computer Vision (WACV), 2022, pp. 587-596, doi: 10.1109/WACV51458.2022.00066. https://ieeexplore.ieee.org/document/9706752


<a id="5">[5]</a> 
H. Su, S. Gong and X. Zhu, "WebLogo-2M: Scalable Logo Detection by Deep Learning from the Web," 2017 IEEE International Conference on Computer Vision Workshops (ICCVW), 2017, pp. 270-279, doi: 10.1109/ICCVW.2017.41. https://ieeexplore.ieee.org/abstract/document/8265251

<a id="6">[6]</a> 
Hou, S., Li, J., Min, W., Hou, Q., Zhao, Y., Zheng, Y., & Jiang, S. (2022). Deep Learning for Logo Detection: A Survey. ArXiv, abs/2210.04399. https://arxiv.org/abs/2210.04399


<!-- <div id="refs"></div> -->


<!--# Contribute
[(Back to top)](#table-of-contents)-->

<!-- This is where you can let people know how they can **contribute** to your project. Some of the ways are given below.

Also this shows how you can add subsections within a section. -->

<!--### Sponsor
[(Back to top)](#table-of-contents)-->

<!-- Your project is gaining traction and it is being used by thousands of people(***with this README there will be even more***). Now it would be a good time to look for people or organisations to sponsor your project. This could be because you are not generating any revenue from your project and you require money for keeping the project alive.

You could add how people can sponsor your project in this section. Add your patreon or GitHub sponsor link here for easy access.

A good idea is to also display the sponsors with their organisation logos or badges to show them your love!(*Someday I will get a sponsor and I can show my love*) -->

<!--### Adding new features or fixing bugs
[(Back to top)](#table-of-contents) -->

<!-- This is to give people an idea how they can raise issues or feature requests in your projects. 

You could also give guidelines for submitting and issue or a pull request to your project.

Personally and by standard, you should use a [issue template](https://github.com/navendu-pottekkat/nsfw-filter/blob/master/ISSUE_TEMPLATE.md) and a [pull request template](https://github.com/navendu-pottekkat/nsfw-filter/blob/master/PULL_REQ_TEMPLATE.md)(click for examples) so that when a user opens a new issue they could easily format it as per your project guidelines.

You could also add contact details for people to get in touch with you regarding your project. -->

<!--# License
[(Back to top)](#table-of-contents)-->

<!-- Adding the license to README is a good practice so that people can easily refer to it.

Make sure you have added a LICENSE file in your project folder. **Shortcut:** Click add new file in your root of your repo in GitHub > Set file name to LICENSE > GitHub shows LICENSE templates > Choose the one that best suits your project!

I personally add the name of the license and provide a link to it like below. -->

<!--[GNU General Public License version 3](https://opensource.org/licenses/GPL-3.0) -->

<!--# Footer
[(Back to top)](#table-of-contents) -->

<!-- Let's also add a footer because I love footers and also you **can** use this to convey important info.

Let's make it an image because by now you have realised that multimedia in images == cool(*please notice the subtle programming joke). -->

<!-- Leave a star in GitHub, give a clap in Medium and share this guide if you found this helpful. -->

<!-- Add the footer here -->

<!-- ![Footer](https://github.com/navendu-pottekkat/awesome-readme/blob/master/fooooooter.png) -->

