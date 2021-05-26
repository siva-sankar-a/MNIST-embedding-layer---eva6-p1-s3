# EVA6-Phase1 Session 3 Submission

Submitted by: _Siva Sankar Anil Kumar_

# Problem statement

- The problem statement requires us to take in an MNIST hand written image and a normal digit as inputs to a neural network and predict the **MNIST digit label** and **the sum of the MNIST digit and the input digit**

### Inputs
  - _MNIST Image_
  - _A normal number_

### Outputs
  - _MNIST output label_
  - _Sum of MNIST number and input digit_

![Problem Statement](assign.png)

# Results

| _Metric_ | _Value_ |
| --- | --- |
| **MNIST Label Accuracy** | **99.87** |
| **MNIST Label + Sum Accuracy** | **87.82** |
| **Epochs** | **30** |
| **Optimizer** | **SGD** |
| **No of parameters** | **6670** |

## Network architecture

### The strategy

- **Convolutional part**
  - The convolutional part of the network consists of 3 identical blocks
  - Each block has the following structure
    <img src="conv.png" width="1200" height="600">
  - The complete network consists of 3 such blocks
  - The output of these layers would end up at **7 X 7**
  - Each layer has kernel size of **3 X 3** with padding **1**
 
- **Output GAP layer**
  - The GAP layer provides a head for the network that predicts the MNIST labels.

- **Sum input embedding**
  - Since the second input is a categorical between 0 and 9, it makes sense to encode the input using an embedding layer
  - Rather than assuming one-hot encoding to be the best encoding style for the problem, the experiment gives 
    an opportunity for the network to find an optimum representation for the input numbers.

- **GAP output embedding**
  - This approach uses the output of the GAP layer to select an index that wins as the same way we implement Binary Cross Entropy Loss
  - Once the winning node is calculated, an embedding for that winning node is computed
  - This embedding is summed with the input embedding to create the next head.
  
- **Output heads**
  - _Head 1_ :  The GAP layer head that predicts MNIST labels
  - _Head 2_ :  Sum of embeddings from winning GAP layer output and the input number which is used to predict the required sum 


### Network Architecture
  - Overall structure
  <img src="network architecture.png" width="1200" height="480">
  
  - Parameter count
```
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1             [2, 8, 28, 28]              72
       BatchNorm2d-2             [2, 8, 28, 28]              16
         MaxPool2d-3             [2, 8, 14, 14]               0
       BatchNorm2d-4             [2, 8, 14, 14]              16
              ReLU-5             [2, 8, 14, 14]               0
            Conv2d-6            [2, 16, 14, 14]           1,152
       BatchNorm2d-7            [2, 16, 14, 14]              32
         MaxPool2d-8              [2, 16, 7, 7]               0
              ReLU-9              [2, 16, 7, 7]               0
           Conv2d-10              [2, 32, 7, 7]           4,608
      BatchNorm2d-11              [2, 32, 7, 7]              64
        MaxPool2d-12              [2, 32, 3, 3]               0
             ReLU-13              [2, 32, 3, 3]               0
AdaptiveAvgPool2d-14              [2, 32, 1, 1]               0
           Linear-15                    [2, 10]             330
        Embedding-16                    [2, 19]             190
        Embedding-17                    [2, 19]             190
================================================================
Total params: 6,670
Trainable params: 6,670
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.01
Forward/backward pass size (MB): 0.44
Params size (MB): 0.03
Estimated Total Size (MB): 0.47
----------------------------------------------------------------
```
### Loss Function and Loss Plot

- Loss function
  The loss function for the problem statement is computed as the sum of cross-entropy losses for the MNIST label prediction and Sum prediction
  
  <img src="loss.gif">

- Loss plot
<img src="loss_plot.png" width="1200" height="480">

### Statistical metrics

- This plot shows the evolution of precision, recall and f1-score of the different MNIST classes on the test dataset while training
  <img src="metrics-test-mnist-labels.png" width="1200" height="480">

- This plot shows the evolution of precision, recall and f1-score of the different Sum label classes on the test dataset while training
  <img src="metrics-test-sum.png" width="1200" height="480">

# Inferences

- This experiment gives good results for adding an embedding layer after the GAP layer from the convolutional layer based on the winning neuron
- Embeddings have been added to produce the final result and can be replaced with concatenation and downsampling as future work
- The neural network tends to have difficulty in classifying sum labels that can occur as a combination of different numbers
  - For example 0 as sum result can occur only with both inputs being 0. THis case is easily learnt by the network
  - Considering 13 which has a low score in statistical metrics can occur as a result of multiple combinations like 9 + 4, 10 + 3, 8 + 5 and so on.

# Training logs

```
TRAIN : epoch=0 acc_l: 92.59 acc_o: 34.03 loss: 2.23697: 100%|██████████| 1875/1875 [00:34<00:00, 53.85it/s]


TEST : epoch=0 acc_l: 0.31 acc_o: 0.17 loss: 0.00539:   0%|          | 0/1875 [00:00<?, ?it/s]0                    : {'precision': 0.9300233255581473, 'recall': 0.9424278237379706, 'f1-score': 0.9361844863731655, 'support': 5923}
1                    : {'precision': 0.958005249343832, 'recall': 0.9744882824087808, 'f1-score': 0.9661764705882353, 'support': 6742}
2                    : {'precision': 0.9107498341074983, 'recall': 0.9214501510574018, 'f1-score': 0.91606874687135, 'support': 5958}
3                    : {'precision': 0.923391048676903, 'recall': 0.9220355570053825, 'f1-score': 0.9227128050273403, 'support': 6131}
4                    : {'precision': 0.9480864635010631, 'recall': 0.9159534406025334, 'f1-score': 0.9317429914678739, 'support': 5842}
5                    : {'precision': 0.888949572182778, 'recall': 0.9007563180225051, 'f1-score': 0.8948140003665018, 'support': 5421}
6                    : {'precision': 0.9448370970522324, 'recall': 0.9261574856370396, 'f1-score': 0.9354040447137129, 'support': 5918}
7                    : {'precision': 0.9457802162336615, 'recall': 0.9355147645650439, 'f1-score': 0.9406194832290163, 'support': 6265}
8                    : {'precision': 0.8939241356159785, 'recall': 0.9102717484190737, 'f1-score': 0.9020238800914556, 'support': 5851}
9                    : {'precision': 0.9087671697473292, 'recall': 0.900823667843335, 'f1-score': 0.904777984129664, 'support': 5949}
accuracy             : 0.9259333333333334
macro avg            : {'precision': 0.9252514112019423, 'recall': 0.9249879239299068, 'f1-score': 0.9250524892858316, 'support': 60000}
weighted avg         : {'precision': 0.9261023222559764, 'recall': 0.9259333333333334, 'f1-score': 0.9259511613361067, 'support': 60000}
0                    : {'precision': 0.5269086357947435, 'recall': 0.7208904109589042, 'f1-score': 0.6088214027476501, 'support': 584}
1                    : {'precision': 0.5590062111801242, 'recall': 0.6623058053965658, 'f1-score': 0.6062874251497006, 'support': 1223}
2                    : {'precision': 0.40803765387400437, 'recall': 0.5909805977975878, 'f1-score': 0.4827586206896552, 'support': 1907}
3                    : {'precision': 0.3414318108195659, 'recall': 0.4192521877486078, 'f1-score': 0.3763613640421354, 'support': 2514}
4                    : {'precision': 0.26918047079337404, 'recall': 0.3971061093247588, 'f1-score': 0.32086256170433886, 'support': 3110}
5                    : {'precision': 0.22106598984771575, 'recall': 0.2403421633554084, 'f1-score': 0.2303014278159704, 'support': 3624}
6                    : {'precision': 0.31866116460339294, 'recall': 0.32798489853704577, 'f1-score': 0.32325581395348835, 'support': 4238}
7                    : {'precision': 0.29235041656147437, 'recall': 0.2384187770228536, 'f1-score': 0.2626445906101157, 'support': 4857}
8                    : {'precision': 0.43431952662721895, 'recall': 0.27887537993920974, 'f1-score': 0.33965756594169366, 'support': 5264}
9                    : {'precision': 0.4371822803195352, 'recall': 0.30709063084509436, 'f1-score': 0.3607670795045944, 'support': 5881}
10                   : {'precision': 0.4258271077908218, 'recall': 0.22769640479360853, 'f1-score': 0.2967278135845315, 'support': 5257}
11                   : {'precision': 0.30965853658536585, 'recall': 0.3221026994114065, 'f1-score': 0.3157580580978909, 'support': 4927}
12                   : {'precision': 0.3113930110100527, 'recall': 0.31042710570269627, 'f1-score': 0.31090930816107065, 'support': 4191}
13                   : {'precision': 0.22760826771653545, 'recall': 0.26421022564981433, 'f1-score': 0.24454725710508923, 'support': 3501}
14                   : {'precision': 0.21468058968058967, 'recall': 0.24389392882065597, 'f1-score': 0.22835674616138518, 'support': 2866}
15                   : {'precision': 0.3323353293413174, 'recall': 0.41195876288659794, 'f1-score': 0.36788805008285763, 'support': 2425}
16                   : {'precision': 0.3708865660085172, 'recall': 0.5195227765726681, 'f1-score': 0.4327987350350124, 'support': 1844}
17                   : {'precision': 0.5609181871689229, 'recall': 0.8048986486486487, 'f1-score': 0.6611168921262573, 'support': 1184}
18                   : {'precision': 0.5592185592185592, 'recall': 0.7595356550580431, 'f1-score': 0.6441631504922644, 'support': 603}
accuracy             : 0.3402833333333333
macro avg            : {'precision': 0.37477212183904374, 'recall': 0.4235522720247461, 'f1-score': 0.3902096770003001, 'support': 60000}
weighted avg         : {'precision': 0.3477753977148187, 'recall': 0.3402833333333333, 'f1-score': 0.3357045397900949, 'support': 60000}
TEST : epoch=0 acc_l: 97.65 acc_o: 49.23 loss: 1.69091: 100%|██████████| 1875/1875 [00:24<00:00, 75.99it/s]


0                    : {'precision': 0.9980451395059534, 'recall': 0.9481681580280263, 'f1-score': 0.9724675324675324, 'support': 5923}
1                    : {'precision': 0.981302950628104, 'recall': 0.996440225452388, 'f1-score': 0.9888136591109803, 'support': 6742}
2                    : {'precision': 0.9933890048712596, 'recall': 0.9583752937227258, 'f1-score': 0.9755680847428669, 'support': 5958}
3                    : {'precision': 0.9790686354048354, 'recall': 0.984178763660088, 'f1-score': 0.9816170489669758, 'support': 6131}
4                    : {'precision': 0.9746450304259635, 'recall': 0.9869907565902089, 'f1-score': 0.9807790440551114, 'support': 5842}
5                    : {'precision': 0.9647101188332733, 'recall': 0.9883785279468733, 'f1-score': 0.9764009111617312, 'support': 5421}
6                    : {'precision': 0.9384468186892043, 'recall': 0.9944237918215614, 'f1-score': 0.9656247436212979, 'support': 5918}
7                    : {'precision': 0.9896856581532416, 'recall': 0.9648842777334398, 'f1-score': 0.9771276165844985, 'support': 6265}
8                    : {'precision': 0.9793796569052158, 'recall': 0.9659887198769441, 'f1-score': 0.9726381001548786, 'support': 5851}
9                    : {'precision': 0.9679786524349566, 'recall': 0.9756261556564129, 'f1-score': 0.971787358727501, 'support': 5949}
accuracy             : 0.9764666666666667
macro avg            : {'precision': 0.9766651665852008, 'recall': 0.9763454670488668, 'f1-score': 0.9762824099593373, 'support': 60000}
weighted avg         : {'precision': 0.9769196911380474, 'recall': 0.9764666666666667, 'f1-score': 0.9764726074525413, 'support': 60000}
0                    : {'precision': 0.9965397923875432, 'recall': 0.9427168576104746, 'f1-score': 0.9688814129520605, 'support': 611}
1                    : {'precision': 0.9875583203732504, 'recall': 0.5300500834724541, 'f1-score': 0.6898424769147202, 'support': 1198}
2                    : {'precision': 0.39861523244312563, 'recall': 0.660655737704918, 'f1-score': 0.4972239358420728, 'support': 1830}
3                    : {'precision': 0.2503617945007236, 'recall': 0.47752365930599366, 'f1-score': 0.32849586328495867, 'support': 2536}
4                    : {'precision': 0.2312683131017162, 'recall': 0.36833333333333335, 'f1-score': 0.28413473900745695, 'support': 3000}
5                    : {'precision': 0.9982698961937716, 'recall': 0.1609483960948396, 'f1-score': 0.2772039394667307, 'support': 3585}
6                    : {'precision': 0.4416072099136312, 'recall': 0.5608011444921316, 'f1-score': 0.4941176470588235, 'support': 4194}
7                    : {'precision': 0.5059139784946236, 'recall': 0.3858929669879024, 'f1-score': 0.437827149005467, 'support': 4877}
8                    : {'precision': 0.7809337251061051, 'recall': 0.4410029498525074, 'f1-score': 0.5636856368563686, 'support': 5424}
9                    : {'precision': 0.8314049586776859, 'recall': 0.49703557312252966, 'f1-score': 0.6221397649969079, 'support': 6072}
10                   : {'precision': 0.7345904298459043, 'recall': 0.6640395894428153, 'f1-score': 0.6975356180207932, 'support': 5456}
11                   : {'precision': 0.3256733848182332, 'recall': 0.36670201484623544, 'f1-score': 0.3449720670391062, 'support': 4715}
12                   : {'precision': 0.4877637130801688, 'recall': 0.5602132299491156, 'f1-score': 0.5214841547310252, 'support': 4127}
13                   : {'precision': 0.47578589634664403, 'recall': 0.3185437997724687, 'f1-score': 0.38160136286201024, 'support': 3516}
14                   : {'precision': 0.39080856945404285, 'recall': 0.3873287671232877, 'f1-score': 0.3890608875128999, 'support': 2920}
15                   : {'precision': 0.36237006237006236, 'recall': 0.7420178799489144, 'f1-score': 0.4869395166922755, 'support': 2349}
16                   : {'precision': 0.49609856262833674, 'recall': 0.6561651276480174, 'f1-score': 0.5650140318054256, 'support': 1841}
17                   : {'precision': 0.6532120523024446, 'recall': 0.982051282051282, 'f1-score': 0.7845681119836121, 'support': 1170}
18                   : {'precision': 0.9707401032702238, 'recall': 0.9740932642487047, 'f1-score': 0.9724137931034482, 'support': 579}
accuracy             : 0.4922666666666667
macro avg            : {'precision': 0.5957639997530652, 'recall': 0.5619008240530488, 'f1-score': 0.5424811636387453, 'support': 60000}
weighted avg         : {'precision': 0.5748971159016005, 'recall': 0.4922666666666667, 'f1-score': 0.4962912399970377, 'support': 60000}
TRAIN : epoch=1 acc_l: 97.75 acc_o: 47.78 loss: 1.60028: 100%|██████████| 1875/1875 [00:34<00:00, 53.91it/s]


TEST : epoch=1 acc_l: 0.74 acc_o: 0.45 loss: 0.01091:   0%|          | 9/1875 [00:00<00:22, 83.00it/s]0                    : {'precision': 0.984427894380501, 'recall': 0.9819348303224718, 'f1-score': 0.983179781928831, 'support': 5923}
1                    : {'precision': 0.9860864416814683, 'recall': 0.9881340848412934, 'f1-score': 0.9871092013631649, 'support': 6742}
2                    : {'precision': 0.9759946281685412, 'recall': 0.9758308157099698, 'f1-score': 0.975912715065044, 'support': 5958}
3                    : {'precision': 0.9813176007866273, 'recall': 0.9766759093133257, 'f1-score': 0.9789912531676612, 'support': 6131}
4                    : {'precision': 0.9814305364511692, 'recall': 0.9770626497774735, 'f1-score': 0.979241722422371, 'support': 5842}
5                    : {'precision': 0.971976401179941, 'recall': 0.9725142962553035, 'f1-score': 0.9722452743199631, 'support': 5421}
6                    : {'precision': 0.9799798115746972, 'recall': 0.9842852314971274, 'f1-score': 0.9821278030686226, 'support': 5918}
7                    : {'precision': 0.9760621433100825, 'recall': 0.9827613727055068, 'f1-score': 0.979400302234948, 'support': 6265}
8                    : {'precision': 0.9675102599179206, 'recall': 0.9670141856092975, 'f1-score': 0.9672621591589025, 'support': 5851}
9                    : {'precision': 0.9681657402728651, 'recall': 0.9662128088754413, 'f1-score': 0.9671882887430591, 'support': 5949}
accuracy             : 0.9774666666666667
macro avg            : {'precision': 0.9772951457723813, 'recall': 0.9772426184907212, 'f1-score': 0.9772658501472569, 'support': 60000}
weighted avg         : {'precision': 0.9774677841737847, 'recall': 0.9774666666666667, 'f1-score': 0.9774641407166109, 'support': 60000}
0                    : {'precision': 0.9847328244274809, 'recall': 0.9791271347248577, 'f1-score': 0.9819219790675547, 'support': 527}
1                    : {'precision': 0.737061273051755, 'recall': 0.9501533742331288, 'f1-score': 0.8301507537688442, 'support': 1304}
2                    : {'precision': 0.5282555282555282, 'recall': 0.6916890080428955, 'f1-score': 0.5990248432783839, 'support': 1865}
3                    : {'precision': 0.4325358851674641, 'recall': 0.5478787878787879, 'f1-score': 0.4834224598930481, 'support': 2475}
4                    : {'precision': 0.4474613087157209, 'recall': 0.531784446595676, 'f1-score': 0.4859923326452374, 'support': 3099}
5                    : {'precision': 0.35243757431629014, 'recall': 0.4038147138964578, 'f1-score': 0.3763809523809524, 'support': 3670}
6                    : {'precision': 0.37228319345814503, 'recall': 0.40696306751352623, 'f1-score': 0.38885142728703076, 'support': 4251}
7                    : {'precision': 0.44330877109579275, 'recall': 0.389515455304929, 'f1-score': 0.41467481934408, 'support': 4788}
8                    : {'precision': 0.5606986899563319, 'recall': 0.3629169022046354, 'f1-score': 0.44063143445435826, 'support': 5307}
9                    : {'precision': 0.6857908847184987, 'recall': 0.43078477601886156, 'f1-score': 0.5291683905668184, 'support': 5938}
10                   : {'precision': 0.5588095238095238, 'recall': 0.4331057390662484, 'f1-score': 0.48799251481442973, 'support': 5419}
11                   : {'precision': 0.44206946805926645, 'recall': 0.3844528939585974, 'f1-score': 0.41125296576658, 'support': 4734}
12                   : {'precision': 0.41962922573609596, 'recall': 0.4629451395572666, 'f1-score': 0.4402242306372269, 'support': 4156}
13                   : {'precision': 0.3929660023446659, 'recall': 0.47184684684684686, 'f1-score': 0.42880900601253674, 'support': 3552}
14                   : {'precision': 0.3609501738122827, 'recall': 0.42788461538461536, 'f1-score': 0.39157762413576364, 'support': 2912}
15                   : {'precision': 0.41923223109083996, 'recall': 0.47811009969657564, 'f1-score': 0.44673957067638725, 'support': 2307}
16                   : {'precision': 0.448796263025512, 'recall': 0.6577145866245392, 'f1-score': 0.5335326783425886, 'support': 1899}
17                   : {'precision': 0.6601539372409709, 'recall': 0.9260797342192691, 'f1-score': 0.770826132042862, 'support': 1204}
18                   : {'precision': 0.9830220713073005, 'recall': 0.9763912310286678, 'f1-score': 0.9796954314720812, 'support': 593}
accuracy             : 0.47781666666666667
macro avg            : {'precision': 0.5384313068204981, 'recall': 0.5743767659366518, 'f1-score': 0.5484668182414086, 'support': 60000}
weighted avg         : {'precision': 0.49129905182840206, 'recall': 0.47781666666666667, 'f1-score': 0.4759545401640356, 'support': 60000}
TEST : epoch=1 acc_l: 98.70 acc_o: 60.41 loss: 1.47870: 100%|██████████| 1875/1875 [00:25<00:00, 74.51it/s]


0                    : {'precision': 0.9856235372785022, 'recall': 0.9954414992402498, 'f1-score': 0.9905081898362031, 'support': 5923}
1                    : {'precision': 0.991566799822459, 'recall': 0.9940670424206467, 'f1-score': 0.9928153470113326, 'support': 6742}
2                    : {'precision': 0.9880551816958277, 'recall': 0.9857334676065794, 'f1-score': 0.9868929591665266, 'support': 5958}
3                    : {'precision': 0.9924651924651925, 'recall': 0.9882564018920241, 'f1-score': 0.9903563255966001, 'support': 6131}
4                    : {'precision': 0.9916854321843063, 'recall': 0.9799726121191373, 'f1-score': 0.9857942315970728, 'support': 5842}
5                    : {'precision': 0.9878026242838662, 'recall': 0.9859804464121011, 'f1-score': 0.986890694239291, 'support': 5421}
6                    : {'precision': 0.9915469146238377, 'recall': 0.9910442717134167, 'f1-score': 0.9912955294515339, 'support': 5918}
7                    : {'precision': 0.9923664122137404, 'recall': 0.9752593774940144, 'f1-score': 0.9837385284173241, 'support': 6265}
8                    : {'precision': 0.977030906941395, 'recall': 0.9887198769441121, 'f1-score': 0.9828406388039416, 'support': 5851}
9                    : {'precision': 0.9711586275484834, 'recall': 0.9848714069591528, 'f1-score': 0.9779669504256385, 'support': 5949}
accuracy             : 0.9869833333333333
macro avg            : {'precision': 0.9869301629057612, 'recall': 0.9869346402801433, 'f1-score': 0.9869099394545465, 'support': 60000}
weighted avg         : {'precision': 0.987035224822966, 'recall': 0.9869833333333333, 'f1-score': 0.9869867184151857, 'support': 60000}
0                    : {'precision': 0.9763912310286678, 'recall': 0.9914383561643836, 'f1-score': 0.9838572642310961, 'support': 584}
1                    : {'precision': 0.627092050209205, 'recall': 0.9925496688741722, 'f1-score': 0.7685897435897436, 'support': 1208}
2                    : {'precision': 0.645002730748225, 'recall': 0.6281914893617021, 'f1-score': 0.6364861223389922, 'support': 1880}
3                    : {'precision': 0.5856950067476383, 'recall': 0.7248434237995824, 'f1-score': 0.647882067549916, 'support': 2395}
4                    : {'precision': 0.7220930232558139, 'recall': 0.6021331609566903, 'f1-score': 0.6566795911173775, 'support': 3094}
5                    : {'precision': 0.5423921271763815, 'recall': 0.7965536409116175, 'f1-score': 0.6453501463634317, 'support': 3598}
6                    : {'precision': 0.44915099507029393, 'recall': 0.5812854442344045, 'f1-score': 0.506746317849418, 'support': 4232}
7                    : {'precision': 0.5804195804195804, 'recall': 0.35834703947368424, 'f1-score': 0.4431168170840219, 'support': 4864}
8                    : {'precision': 0.714527027027027, 'recall': 0.5473197781885397, 'f1-score': 0.6198450910613356, 'support': 5410}
9                    : {'precision': 0.8472514619883041, 'recall': 0.6054831160147108, 'f1-score': 0.7062493906600371, 'support': 5982}
10                   : {'precision': 0.7418807047336022, 'recall': 0.6552305961754781, 'f1-score': 0.6958685913389746, 'support': 5334}
11                   : {'precision': 0.6270116113261357, 'recall': 0.6360818350898946, 'f1-score': 0.6315141567501026, 'support': 4839}
12                   : {'precision': 0.5622286541244573, 'recall': 0.5677057963955188, 'f1-score': 0.5649539505574407, 'support': 4106}
13                   : {'precision': 0.4509374239103969, 'recall': 0.5186222346681602, 'f1-score': 0.48241729617087775, 'support': 3571}
14                   : {'precision': 0.33907056798623064, 'recall': 0.20205128205128206, 'f1-score': 0.2532133676092545, 'support': 2925}
15                   : {'precision': 0.4360576923076923, 'recall': 0.7296862429605793, 'f1-score': 0.5458922660246764, 'support': 2486}
16                   : {'precision': 0.6362637362637362, 'recall': 0.6613363792118789, 'f1-score': 0.6485578269392327, 'support': 1751}
17                   : {'precision': 0.6582207207207207, 'recall': 0.9890016920473773, 'f1-score': 0.7903989181879647, 'support': 1182}
18                   : {'precision': 0.9716814159292035, 'recall': 0.9821109123434705, 'f1-score': 0.9768683274021353, 'support': 559}
accuracy             : 0.6041166666666666
macro avg            : {'precision': 0.6375456716301743, 'recall': 0.6721037941538488, 'f1-score': 0.6423414343592647, 'support': 60000}
weighted avg         : {'precision': 0.6201299841851571, 'recall': 0.6041166666666666, 'f1-score': 0.6001134918349497, 'support': 60000}
TRAIN : epoch=2 acc_l: 98.38 acc_o: 51.15 loss: 1.45090: 100%|██████████| 1875/1875 [00:35<00:00, 53.00it/s]


TEST : epoch=2 acc_l: 0.31 acc_o: 0.15 loss: 0.00479:   0%|          | 0/1875 [00:00<?, ?it/s]0                    : {'precision': 0.9897150564828865, 'recall': 0.991051831841972, 'f1-score': 0.9903829930825038, 'support': 5923}
1                    : {'precision': 0.9912527798369163, 'recall': 0.9916938593889054, 'f1-score': 0.9914732705568325, 'support': 6742}
2                    : {'precision': 0.9809077206498074, 'recall': 0.9830480026854649, 'f1-score': 0.9819766954480676, 'support': 5958}
3                    : {'precision': 0.9857610474631752, 'recall': 0.9823846028380362, 'f1-score': 0.9840699289273752, 'support': 6131}
4                    : {'precision': 0.9854177388917481, 'recall': 0.983224922971585, 'f1-score': 0.9843201096735498, 'support': 5842}
5                    : {'precision': 0.9788252623826184, 'recall': 0.9806308799114555, 'f1-score': 0.9797272392185773, 'support': 5421}
6                    : {'precision': 0.9866396076441738, 'recall': 0.9858060155457925, 'f1-score': 0.9862226354492435, 'support': 5918}
7                    : {'precision': 0.9834552974864779, 'recall': 0.9867517956903432, 'f1-score': 0.9851007887817703, 'support': 6265}
8                    : {'precision': 0.9787671232876712, 'recall': 0.9769270210220475, 'f1-score': 0.9778462064836199, 'support': 5851}
9                    : {'precision': 0.9756056527590848, 'recall': 0.9747856782652546, 'f1-score': 0.9751954931472295, 'support': 5949}
accuracy             : 0.9837833333333333
macro avg            : {'precision': 0.983634728688456, 'recall': 0.9836304610160858, 'f1-score': 0.9836315360768768, 'support': 60000}
weighted avg         : {'precision': 0.9837834164038273, 'recall': 0.9837833333333333, 'f1-score': 0.9837823121591147, 'support': 60000}
0                    : {'precision': 0.9895470383275261, 'recall': 0.9895470383275261, 'f1-score': 0.9895470383275261, 'support': 574}
1                    : {'precision': 0.7315700619020822, 'recall': 0.9916094584286804, 'f1-score': 0.8419689119170986, 'support': 1311}
2                    : {'precision': 0.5228290062667861, 'recall': 0.6424642464246425, 'f1-score': 0.5765054294175715, 'support': 1818}
3                    : {'precision': 0.4586535505687058, 'recall': 0.6020984665052461, 'f1-score': 0.5206770197173268, 'support': 2478}
4                    : {'precision': 0.4118271954674221, 'recall': 0.38484447385837195, 'f1-score': 0.3978788915497776, 'support': 3022}
5                    : {'precision': 0.39170628631801374, 'recall': 0.4157555368657135, 'f1-score': 0.4033727730178159, 'support': 3567}
6                    : {'precision': 0.41556091676718937, 'recall': 0.48992652287271865, 'f1-score': 0.44968998150766887, 'support': 4219}
7                    : {'precision': 0.5035742652899127, 'recall': 0.3979912115505336, 'f1-score': 0.4446002805049089, 'support': 4779}
8                    : {'precision': 0.535728463676062, 'recall': 0.49287801314828344, 'f1-score': 0.5134106905078942, 'support': 5476}
9                    : {'precision': 0.6775887943971985, 'recall': 0.45606060606060606, 'f1-score': 0.5451801167236868, 'support': 5940}
10                   : {'precision': 0.6097417840375586, 'recall': 0.5580021482277121, 'f1-score': 0.5827257431295568, 'support': 5586}
11                   : {'precision': 0.5058823529411764, 'recall': 0.461489898989899, 'f1-score': 0.4826675470452294, 'support': 4752}
12                   : {'precision': 0.4871650211565585, 'recall': 0.42255933447516514, 'f1-score': 0.4525681341719078, 'support': 4087}
13                   : {'precision': 0.41552839683680803, 'recall': 0.48530646515533166, 'f1-score': 0.447714949651433, 'support': 3573}
14                   : {'precision': 0.3732014388489209, 'recall': 0.4237576582709326, 'f1-score': 0.39687599617468916, 'support': 2938}
15                   : {'precision': 0.49112670243499795, 'recall': 0.5138169257340242, 'f1-score': 0.5022156573116692, 'support': 2316}
16                   : {'precision': 0.4859154929577465, 'recall': 0.6854304635761589, 'f1-score': 0.5686813186813185, 'support': 1812}
17                   : {'precision': 0.6594626168224299, 'recall': 0.9567796610169491, 'f1-score': 0.780774550484094, 'support': 1180}
18                   : {'precision': 0.9755671902268761, 'recall': 0.9772727272727273, 'f1-score': 0.9764192139737992, 'support': 572}
accuracy             : 0.51145
macro avg            : {'precision': 0.5601145565917879, 'recall': 0.5972416240400642, 'f1-score': 0.5722881180955248, 'support': 60000}
weighted avg         : {'precision': 0.5185088988762975, 'recall': 0.51145, 'f1-score': 0.5093443863401894, 'support': 60000}
TEST : epoch=2 acc_l: 98.41 acc_o: 53.31 loss: 1.41227: 100%|██████████| 1875/1875 [00:25<00:00, 73.68it/s]


TRAIN : epoch=3 acc_l: 0.16 acc_o: 0.10 loss: 0.00203:   0%|          | 0/1875 [00:00<?, ?it/s]0                    : {'precision': 0.9969024264326277, 'recall': 0.9780516630086105, 'f1-score': 0.9873870802795294, 'support': 5923}
1                    : {'precision': 0.9908635425876805, 'recall': 0.9973301690892911, 'f1-score': 0.994086339444116, 'support': 6742}
2                    : {'precision': 0.9678577255669767, 'recall': 0.995636119503189, 'f1-score': 0.9815504260776041, 'support': 5958}
3                    : {'precision': 0.9962052466589671, 'recall': 0.9848311857771979, 'f1-score': 0.9904855643044619, 'support': 6131}
4                    : {'precision': 0.9604417339706609, 'recall': 0.9974323861691201, 'f1-score': 0.9785876228062809, 'support': 5842}
5                    : {'precision': 0.989979588049731, 'recall': 0.9841357683084302, 'f1-score': 0.9870490286771509, 'support': 5421}
6                    : {'precision': 0.9665244502789629, 'recall': 0.9952686718485975, 'f1-score': 0.9806859806859807, 'support': 5918}
7                    : {'precision': 0.9972057856673241, 'recall': 0.9683958499600958, 'f1-score': 0.982589683375172, 'support': 6265}
8                    : {'precision': 0.9959356776815692, 'recall': 0.9632541445906683, 'f1-score': 0.9793223284100782, 'support': 5851}
9                    : {'precision': 0.9807432432432432, 'recall': 0.9759623466128761, 'f1-score': 0.9783469542505687, 'support': 5949}
accuracy             : 0.9841333333333333
macro avg            : {'precision': 0.9842659420137743, 'recall': 0.9840298304868076, 'f1-score': 0.9840091008310943, 'support': 60000}
weighted avg         : {'precision': 0.9844318812813425, 'recall': 0.9841333333333333, 'f1-score': 0.9841449143608518, 'support': 60000}
0                    : {'precision': 1.0, 'recall': 0.9907063197026023, 'f1-score': 0.9953314659197012, 'support': 538}
1                    : {'precision': 0.6459403905447071, 'recall': 0.988985051140834, 'f1-score': 0.7814734224432701, 'support': 1271}
2                    : {'precision': 0.492089925062448, 'recall': 0.6290580095795636, 'f1-score': 0.5522074281709881, 'support': 1879}
3                    : {'precision': 0.5900156006240249, 'recall': 0.7456624605678234, 'f1-score': 0.6587702490855252, 'support': 2536}
4                    : {'precision': 0.5754593175853019, 'recall': 0.5913688469318948, 'f1-score': 0.583305620219488, 'support': 2966}
5                    : {'precision': 0.3727422003284072, 'recall': 0.3141433711597011, 'f1-score': 0.34094322619405226, 'support': 3613}
6                    : {'precision': 0.33121754183496827, 'recall': 0.2733984281971898, 'f1-score': 0.29954337899543376, 'support': 4199}
7                    : {'precision': 0.3770032051282051, 'recall': 0.3912681912681913, 'f1-score': 0.3840032646398694, 'support': 4810}
8                    : {'precision': 0.4434941967012828, 'recall': 0.539175640549573, 'f1-score': 0.4866767219708396, 'support': 5386}
9                    : {'precision': 0.8368854788292128, 'recall': 0.4919812896759105, 'f1-score': 0.6196738558653341, 'support': 5986}
10                   : {'precision': 0.6998353328628558, 'recall': 0.5424872355944566, 'f1-score': 0.6111967128916281, 'support': 5484}
11                   : {'precision': 0.43727206418672504, 'recall': 0.5080508474576271, 'f1-score': 0.47001176009408074, 'support': 4720}
12                   : {'precision': 0.7441464452958706, 'recall': 0.42864149092692494, 'f1-score': 0.5439551890462113, 'support': 4078}
13                   : {'precision': 0.4128774730996182, 'recall': 0.6604664075513603, 'f1-score': 0.5081161896625374, 'support': 3602}
14                   : {'precision': 0.48737697903294824, 'recall': 0.39127447612504296, 'f1-score': 0.4340701219512196, 'support': 2911}
15                   : {'precision': 0.5917001338688086, 'recall': 0.7403685092127303, 'f1-score': 0.6577380952380953, 'support': 2388}
16                   : {'precision': 0.6109054190508246, 'recall': 0.968, 'f1-score': 0.7490713990920348, 'support': 1875}
17                   : {'precision': 0.9796264855687606, 'recall': 0.48487394957983193, 'f1-score': 0.6486790331646992, 'support': 1190}
18                   : {'precision': 0.9805309734513274, 'recall': 0.9753521126760564, 'f1-score': 0.9779346866725508, 'support': 568}
accuracy             : 0.5330666666666667
macro avg            : {'precision': 0.611006271739805, 'recall': 0.613434875678806, 'f1-score': 0.59487904322724, 'support': 60000}
weighted avg         : {'precision': 0.559322913609199, 'recall': 0.5330666666666667, 'f1-score': 0.5303206993189619, 'support': 60000}
TRAIN : epoch=3 acc_l: 98.41 acc_o: 57.48 loss: 1.37443: 100%|██████████| 1875/1875 [00:35<00:00, 52.75it/s]


TEST : epoch=3 acc_l: 0.31 acc_o: 0.19 loss: 0.00436:   0%|          | 0/1875 [00:00<?, ?it/s]0                    : {'precision': 0.9915497718438397, 'recall': 0.9905453317575553, 'f1-score': 0.9910472972972972, 'support': 5923}
1                    : {'precision': 0.990386037568407, 'recall': 0.9931770987837437, 'f1-score': 0.9917796045323263, 'support': 6742}
2                    : {'precision': 0.9820349227669577, 'recall': 0.9817052702249077, 'f1-score': 0.9818700688265906, 'support': 5958}
3                    : {'precision': 0.9841425535393167, 'recall': 0.9818952862502038, 'f1-score': 0.9830176355323317, 'support': 6131}
4                    : {'precision': 0.9854252400548696, 'recall': 0.983738445737761, 'f1-score': 0.9845811204385815, 'support': 5842}
5                    : {'precision': 0.9793434157137587, 'recall': 0.9795240730492529, 'f1-score': 0.9794337360509084, 'support': 5421}
6                    : {'precision': 0.9880168776371308, 'recall': 0.9891855356539372, 'f1-score': 0.9886008612682597, 'support': 5918}
7                    : {'precision': 0.981404958677686, 'recall': 0.985634477254589, 'f1-score': 0.983515170821056, 'support': 6265}
8                    : {'precision': 0.9794766546947152, 'recall': 0.9788070415313621, 'f1-score': 0.9791417336296803, 'support': 5851}
9                    : {'precision': 0.977575451020064, 'recall': 0.974617582787023, 'f1-score': 0.976094276094276, 'support': 5949}
accuracy             : 0.98405
macro avg            : {'precision': 0.9839355883516745, 'recall': 0.9838830143030336, 'f1-score': 0.9839081504491307, 'support': 60000}
weighted avg         : {'precision': 0.9840474858534428, 'recall': 0.98405, 'f1-score': 0.9840475498672024, 'support': 60000}
0                    : {'precision': 0.9921383647798742, 'recall': 0.9921383647798742, 'f1-score': 0.9921383647798742, 'support': 636}
1                    : {'precision': 0.7523584905660378, 'recall': 0.9929961089494164, 'f1-score': 0.8560885608856089, 'support': 1285}
2                    : {'precision': 0.5812258894127732, 'recall': 0.7178401270513499, 'f1-score': 0.6423495973472287, 'support': 1889}
3                    : {'precision': 0.5036080516521079, 'recall': 0.5447822514379622, 'f1-score': 0.5233866193013618, 'support': 2434}
4                    : {'precision': 0.5305420131918555, 'recall': 0.5946640951462552, 'f1-score': 0.5607759927250682, 'support': 3111}
5                    : {'precision': 0.5149532710280373, 'recall': 0.47609447004608296, 'f1-score': 0.49476204729123013, 'support': 3472}
6                    : {'precision': 0.5229619863766205, 'recall': 0.5590791637303265, 'f1-score': 0.5404178019981835, 'support': 4257}
7                    : {'precision': 0.4870954526833265, 'recall': 0.492135761589404, 'f1-score': 0.4896026353716286, 'support': 4832}
8                    : {'precision': 0.5775225594749795, 'recall': 0.5225459268881054, 'f1-score': 0.5486604968339016, 'support': 5389}
9                    : {'precision': 0.7511690868816145, 'recall': 0.5209080047789725, 'f1-score': 0.6151985486797016, 'support': 5859}
10                   : {'precision': 0.6328778299000611, 'recall': 0.5739918608953015, 'f1-score': 0.6019982539528567, 'support': 5406}
11                   : {'precision': 0.548228115465853, 'recall': 0.4930350358801182, 'f1-score': 0.5191687965329482, 'support': 4738}
12                   : {'precision': 0.5484210526315789, 'recall': 0.6070845956653461, 'f1-score': 0.576263687645172, 'support': 4291}
13                   : {'precision': 0.48779724655819773, 'recall': 0.45678288895399943, 'f1-score': 0.4717809048267514, 'support': 3413}
14                   : {'precision': 0.4949788583509514, 'recall': 0.6169301712779973, 'f1-score': 0.549266862170088, 'support': 3036}
15                   : {'precision': 0.5615062761506276, 'recall': 0.5688851208139042, 'f1-score': 0.5651716150768583, 'support': 2359}
16                   : {'precision': 0.5860486461679669, 'recall': 0.6962922573609597, 'f1-score': 0.6364315973087465, 'support': 1834}
17                   : {'precision': 0.6966794380587484, 'recall': 0.9462272333044233, 'f1-score': 0.8025009194556822, 'support': 1153}
18                   : {'precision': 0.9701986754966887, 'recall': 0.966996699669967, 'f1-score': 0.9685950413223141, 'support': 606}
accuracy             : 0.5748333333333333
macro avg            : {'precision': 0.6179111213067316, 'recall': 0.6494426388536718, 'f1-score': 0.6291872812371161, 'support': 60000}
weighted avg         : {'precision': 0.5797256450738243, 'recall': 0.5748333333333333, 'f1-score': 0.5730953860549596, 'support': 60000}
TEST : epoch=3 acc_l: 99.07 acc_o: 58.59 loss: 1.30766: 100%|██████████| 1875/1875 [00:25<00:00, 72.29it/s]


TRAIN : epoch=4 acc_l: 0.11 acc_o: 0.07 loss: 0.00132:   0%|          | 0/1875 [00:00<?, ?it/s]0                    : {'precision': 0.996776382762131, 'recall': 0.9918959986493331, 'f1-score': 0.9943302022509943, 'support': 5923}
1                    : {'precision': 0.9985023214018272, 'recall': 0.9888757045387125, 'f1-score': 0.9936656978910501, 'support': 6742}
2                    : {'precision': 0.9876625541847283, 'recall': 0.9942933870426317, 'f1-score': 0.9909668785547006, 'support': 5958}
3                    : {'precision': 0.996225795864785, 'recall': 0.9902136682433534, 'f1-score': 0.9932106339468302, 'support': 6131}
4                    : {'precision': 0.9929541158274617, 'recall': 0.9890448476549127, 'f1-score': 0.9909956264471315, 'support': 5842}
5                    : {'precision': 0.9907766094816455, 'recall': 0.9907766094816455, 'f1-score': 0.9907766094816455, 'support': 5421}
6                    : {'precision': 0.9848156182212582, 'recall': 0.9972963839134843, 'f1-score': 0.9910167072454035, 'support': 5918}
7                    : {'precision': 0.9932562620423893, 'recall': 0.987390263367917, 'f1-score': 0.9903145761626512, 'support': 6265}
8                    : {'precision': 0.9778039347570203, 'recall': 0.9938472056058794, 'f1-score': 0.9857602983556535, 'support': 5851}
9                    : {'precision': 0.9875189745319616, 'recall': 0.9841990250462263, 'f1-score': 0.9858562047482743, 'support': 5949}
accuracy             : 0.9907333333333334
macro avg            : {'precision': 0.9906292569075209, 'recall': 0.9907833093544097, 'f1-score': 0.9906893435084333, 'support': 60000}
weighted avg         : {'precision': 0.99077952458363, 'recall': 0.9907333333333334, 'f1-score': 0.9907393865473147, 'support': 60000}
0                    : {'precision': 0.99822695035461, 'recall': 0.9929453262786596, 'f1-score': 0.995579133510168, 'support': 567}
1                    : {'precision': 0.9968798751950078, 'recall': 0.9891640866873065, 'f1-score': 0.9930069930069929, 'support': 1292}
2                    : {'precision': 0.749900911613159, 'recall': 0.9931758530183727, 'f1-score': 0.854561878952123, 'support': 1905}
3                    : {'precision': 0.41586648376771834, 'recall': 0.7421460628314973, 'f1-score': 0.533040293040293, 'support': 2451}
4                    : {'precision': 0.6703488372093023, 'recall': 0.370978120978121, 'f1-score': 0.47763048881524434, 'support': 3108}
5                    : {'precision': 0.6520055325034578, 'recall': 0.6626370536969356, 'f1-score': 0.6572783045175683, 'support': 3557}
6                    : {'precision': 0.47465437788018433, 'recall': 0.5478723404255319, 'f1-score': 0.508641975308642, 'support': 4136}
7                    : {'precision': 0.6564927857935627, 'recall': 0.4844389844389844, 'f1-score': 0.557492931196984, 'support': 4884}
8                    : {'precision': 0.5464069104943944, 'recall': 0.5546641791044776, 'f1-score': 0.5505045829089898, 'support': 5360}
9                    : {'precision': 0.5539004914004914, 'recall': 0.5944297956493079, 'f1-score': 0.5734499205087441, 'support': 6068}
10                   : {'precision': 0.9951259138911455, 'recall': 0.2271041898405636, 'f1-score': 0.369811320754717, 'support': 5394}
11                   : {'precision': 0.5357795371498173, 'recall': 0.7447619047619047, 'f1-score': 0.6232179226069245, 'support': 4725}
12                   : {'precision': 0.5684964200477327, 'recall': 0.5808339429407462, 'f1-score': 0.5745989627306719, 'support': 4101}
13                   : {'precision': 0.43855362814561444, 'recall': 0.5156564205688021, 'f1-score': 0.47398996567203594, 'support': 3481}
14                   : {'precision': 0.6198749131341209, 'recall': 0.6061841658171934, 'f1-score': 0.6129531008417799, 'support': 2943}
15                   : {'precision': 0.6704035874439462, 'recall': 0.49812578092461474, 'f1-score': 0.5715651135005974, 'support': 2401}
16                   : {'precision': 0.496486151302191, 'recall': 0.6595277320153762, 'f1-score': 0.566509433962264, 'support': 1821}
17                   : {'precision': 0.6465373961218837, 'recall': 0.9915038232795242, 'f1-score': 0.7826961770623743, 'support': 1177}
18                   : {'precision': 0.9839486356340289, 'recall': 0.9745627980922098, 'f1-score': 0.9792332268370607, 'support': 629}
accuracy             : 0.5859333333333333
macro avg            : {'precision': 0.6668362810043352, 'recall': 0.6700375032289543, 'f1-score': 0.6450400908281144, 'support': 60000}
weighted avg         : {'precision': 0.6273153115004657, 'recall': 0.5859333333333333, 'f1-score': 0.5758687059225248, 'support': 60000}
TRAIN : epoch=4 acc_l: 98.75 acc_o: 60.43 loss: 1.29944: 100%|██████████| 1875/1875 [00:36<00:00, 52.03it/s]


TEST : epoch=4 acc_l: 0.69 acc_o: 0.43 loss: 0.00864:   0%|          | 8/1875 [00:00<00:26, 70.80it/s]0                    : {'precision': 0.9932398174750718, 'recall': 0.9922336653722775, 'f1-score': 0.9927364864864865, 'support': 5923}
1                    : {'precision': 0.9915643036850673, 'recall': 0.993770394541679, 'f1-score': 0.9926661234165494, 'support': 6742}
2                    : {'precision': 0.9869083585095669, 'recall': 0.9869083585095669, 'f1-score': 0.9869083585095669, 'support': 5958}
3                    : {'precision': 0.9914935383608703, 'recall': 0.988582612950579, 'f1-score': 0.9900359359686377, 'support': 6131}
4                    : {'precision': 0.9874699622382423, 'recall': 0.9847654912701129, 'f1-score': 0.9861158724717175, 'support': 5842}
5                    : {'precision': 0.9841620626151013, 'recall': 0.985795978601734, 'f1-score': 0.9849783430098608, 'support': 5421}
6                    : {'precision': 0.9888607594936709, 'recall': 0.9900304156809733, 'f1-score': 0.9894452419150553, 'support': 5918}
7                    : {'precision': 0.9855187778485042, 'recall': 0.9885075818036712, 'f1-score': 0.987010917204558, 'support': 6265}
8                    : {'precision': 0.9837690073466598, 'recall': 0.9841052811485216, 'f1-score': 0.9839371155160628, 'support': 5851}
9                    : {'precision': 0.9807983830217282, 'recall': 0.9788199697428139, 'f1-score': 0.9798081776880364, 'support': 5949}
accuracy             : 0.9874666666666667
macro avg            : {'precision': 0.9873784970594484, 'recall': 0.9873519749621928, 'f1-score': 0.9873642572186532, 'support': 60000}
weighted avg         : {'precision': 0.9874671679893054, 'recall': 0.9874666666666667, 'f1-score': 0.9874659218547376, 'support': 60000}
0                    : {'precision': 0.9931856899488927, 'recall': 0.9965811965811966, 'f1-score': 0.9948805460750852, 'support': 585}
1                    : {'precision': 0.7876496534341525, 'recall': 0.9928514694201748, 'f1-score': 0.8784258608573436, 'support': 1259}
2                    : {'precision': 0.657184894556155, 'recall': 0.7436182019977803, 'f1-score': 0.6977349648529029, 'support': 1802}
3                    : {'precision': 0.5591795561533288, 'recall': 0.6702942361950827, 'f1-score': 0.6097158570119158, 'support': 2481}
4                    : {'precision': 0.5692605747497578, 'recall': 0.5837748344370861, 'f1-score': 0.576426352787314, 'support': 3020}
5                    : {'precision': 0.5551546391752578, 'recall': 0.5870809484873263, 'f1-score': 0.5706716121340575, 'support': 3669}
6                    : {'precision': 0.5226488848011037, 'recall': 0.5379881656804734, 'f1-score': 0.5302076043853511, 'support': 4225}
7                    : {'precision': 0.5601023017902813, 'recall': 0.5469302809573361, 'f1-score': 0.5534379277666631, 'support': 4805}
8                    : {'precision': 0.5888223552894212, 'recall': 0.5435784042749217, 'f1-score': 0.5652965411516719, 'support': 5427}
9                    : {'precision': 0.7492489022417379, 'recall': 0.5515481456277646, 'f1-score': 0.635374816266536, 'support': 5878}
10                   : {'precision': 0.6913300045392646, 'recall': 0.563448020717721, 'f1-score': 0.6208724011414594, 'support': 5406}
11                   : {'precision': 0.5814249363867684, 'recall': 0.5732803679698933, 'f1-score': 0.5773239288346141, 'support': 4783}
12                   : {'precision': 0.5521262002743484, 'recall': 0.5734979814770839, 'f1-score': 0.5626092020966802, 'support': 4211}
13                   : {'precision': 0.5358961303462322, 'recall': 0.5914582747962911, 'f1-score': 0.5623080005342593, 'support': 3559}
14                   : {'precision': 0.5401974612129761, 'recall': 0.5312066574202496, 'f1-score': 0.5356643356643357, 'support': 2884}
15                   : {'precision': 0.5662051828186013, 'recall': 0.6544932293803857, 'f1-score': 0.6071564522268746, 'support': 2437}
16                   : {'precision': 0.5895196506550219, 'recall': 0.687995469988675, 'f1-score': 0.6349621113143454, 'support': 1766}
17                   : {'precision': 0.6839953271028038, 'recall': 0.9848612279226241, 'f1-score': 0.8073078248879697, 'support': 1189}
18                   : {'precision': 0.9769736842105263, 'recall': 0.9674267100977199, 'f1-score': 0.972176759410802, 'support': 614}
accuracy             : 0.60435
macro avg            : {'precision': 0.6452687384045596, 'recall': 0.6779954643910413, 'f1-score': 0.6575027947052727, 'support': 60000}
weighted avg         : {'precision': 0.6090318551031215, 'recall': 0.60435, 'f1-score': 0.6030897540619224, 'support': 60000}
TEST : epoch=4 acc_l: 98.43 acc_o: 60.50 loss: 1.29675: 100%|██████████| 1875/1875 [00:26<00:00, 71.18it/s]


TRAIN : epoch=5 acc_l: 0.05 acc_o: 0.03 loss: 0.00069:   0%|          | 0/1875 [00:00<?, ?it/s]0                    : {'precision': 0.9950814111261872, 'recall': 0.9905453317575553, 'f1-score': 0.9928081902022168, 'support': 5923}
1                    : {'precision': 0.98376480912681, 'recall': 0.9976268169682587, 'f1-score': 0.9906473230723912, 'support': 6742}
2                    : {'precision': 0.9818030050083473, 'recall': 0.9870762000671366, 'f1-score': 0.9844325410110478, 'support': 5958}
3                    : {'precision': 0.9986743993371997, 'recall': 0.983037024955146, 'f1-score': 0.9907940161104719, 'support': 6131}
4                    : {'precision': 0.9436893203883495, 'recall': 0.9982882574460801, 'f1-score': 0.9702212610214607, 'support': 5842}
5                    : {'precision': 0.9979249198264478, 'recall': 0.975834716841911, 'f1-score': 0.9867562022010818, 'support': 5421}
6                    : {'precision': 0.9824737105658488, 'recall': 0.9945927678269686, 'f1-score': 0.9884960953900412, 'support': 5918}
7                    : {'precision': 0.994343891402715, 'recall': 0.982122905027933, 'f1-score': 0.9881956155143339, 'support': 6265}
8                    : {'precision': 0.9931010083141695, 'recall': 0.959494103572039, 'f1-score': 0.976008344923505, 'support': 5851}
9                    : {'precision': 0.9755439365828976, 'recall': 0.9722642460917801, 'f1-score': 0.9739013301902676, 'support': 5949}
accuracy             : 0.9843333333333333
macro avg            : {'precision': 0.9846400411678973, 'recall': 0.9840882370554809, 'f1-score': 0.9842260919636818, 'support': 60000}
weighted avg         : {'precision': 0.9846606227440323, 'recall': 0.9843333333333333, 'f1-score': 0.9843620266170541, 'support': 60000}
0                    : {'precision': 0.9951690821256038, 'recall': 0.9903846153846154, 'f1-score': 0.9927710843373494, 'support': 624}
1                    : {'precision': 0.9937743190661479, 'recall': 0.9960998439937597, 'f1-score': 0.9949357226334242, 'support': 1282}
2                    : {'precision': 0.667379679144385, 'recall': 0.6823400765445599, 'f1-score': 0.6747769667477697, 'support': 1829}
3                    : {'precision': 0.7647537647537648, 'recall': 0.7537103890894504, 'f1-score': 0.7591919191919191, 'support': 2493}
4                    : {'precision': 0.7007042253521126, 'recall': 0.9933444259567388, 'f1-score': 0.8217481073640743, 'support': 3005}
5                    : {'precision': 0.4496606334841629, 'recall': 0.6628682601445247, 'f1-score': 0.5358346439002472, 'support': 3598}
6                    : {'precision': 0.5741335044929397, 'recall': 0.4282978214029208, 'f1-score': 0.4906074317839024, 'support': 4177}
7                    : {'precision': 0.48457792207792205, 'recall': 0.37034739454094295, 'f1-score': 0.41983122362869196, 'support': 4836}
8                    : {'precision': 0.5490676210209079, 'recall': 0.5337850210584142, 'f1-score': 0.5413184772516249, 'support': 5461}
9                    : {'precision': 0.7308560870457812, 'recall': 0.596614714261773, 'f1-score': 0.6569477763424986, 'support': 5967}
10                   : {'precision': 0.5, 'recall': 0.34405802492095966, 'f1-score': 0.4076236642062355, 'support': 5377}
11                   : {'precision': 0.462289029535865, 'recall': 0.7456401531263293, 'f1-score': 0.5707309132345759, 'support': 4702}
12                   : {'precision': 0.7561501942166595, 'recall': 0.4212551094012984, 'f1-score': 0.5410747374922792, 'support': 4159}
13                   : {'precision': 0.5547961348102758, 'recall': 0.6645962732919255, 'f1-score': 0.6047527296082209, 'support': 3542}
14                   : {'precision': 0.5916398713826366, 'recall': 0.5738045738045738, 'f1-score': 0.5825857519788918, 'support': 2886}
15                   : {'precision': 0.7457697069748246, 'recall': 0.7384552513281569, 'f1-score': 0.7420944558521559, 'support': 2447}
16                   : {'precision': 0.6663037561241154, 'recall': 0.6666666666666666, 'f1-score': 0.6664851619929214, 'support': 1836}
17                   : {'precision': 0.6633892423366108, 'recall': 0.9590301003344481, 'f1-score': 0.7842735042735043, 'support': 1196}
18                   : {'precision': 0.9671848013816926, 'recall': 0.9605488850771869, 'f1-score': 0.963855421686747, 'support': 583}
accuracy             : 0.60505
macro avg            : {'precision': 0.6746105039645477, 'recall': 0.6885182947541708, 'f1-score': 0.6711284049214228, 'support': 60000}
weighted avg         : {'precision': 0.6155300938975351, 'recall': 0.60505, 'f1-score': 0.5977233401318627, 'support': 60000}
TRAIN : epoch=5 acc_l: 98.85 acc_o: 63.99 loss: 1.24939: 100%|██████████| 1875/1875 [00:36<00:00, 51.65it/s]


TEST : epoch=5 acc_l: 0.32 acc_o: 0.23 loss: 0.00366:   0%|          | 0/1875 [00:00<?, ?it/s]0                    : {'precision': 0.9922506738544474, 'recall': 0.9944284990714165, 'f1-score': 0.9933383927818534, 'support': 5923}
1                    : {'precision': 0.9925969795676636, 'recall': 0.9943636902996144, 'f1-score': 0.9934795494961469, 'support': 6742}
2                    : {'precision': 0.9874097700184656, 'recall': 0.9872440416247062, 'f1-score': 0.9873268988669743, 'support': 5958}
3                    : {'precision': 0.9916693890885332, 'recall': 0.9902136682433534, 'f1-score': 0.9909409940422754, 'support': 6131}
4                    : {'precision': 0.986981843096951, 'recall': 0.9863060595686409, 'f1-score': 0.9866438356164383, 'support': 5842}
5                    : {'precision': 0.9870753323485968, 'recall': 0.9861649142224682, 'f1-score': 0.9866199132601274, 'support': 5421}
6                    : {'precision': 0.990537343697195, 'recall': 0.990537343697195, 'f1-score': 0.990537343697195, 'support': 5918}
7                    : {'precision': 0.9885149146594353, 'recall': 0.989146049481245, 'f1-score': 0.9888303813626934, 'support': 6265}
8                    : {'precision': 0.986810551558753, 'recall': 0.9846180140146984, 'f1-score': 0.9857130635640347, 'support': 5851}
9                    : {'precision': 0.980510752688172, 'recall': 0.9810052109598252, 'f1-score': 0.9807579195025627, 'support': 5949}
accuracy             : 0.9885166666666667
macro avg            : {'precision': 0.9884357550578212, 'recall': 0.9884027491183163, 'f1-score': 0.9884188292190302, 'support': 60000}
weighted avg         : {'precision': 0.9885153043507433, 'recall': 0.9885166666666667, 'f1-score': 0.9885155582633024, 'support': 60000}
0                    : {'precision': 0.9916943521594684, 'recall': 0.9916943521594684, 'f1-score': 0.9916943521594684, 'support': 602}
1                    : {'precision': 0.8717406624383368, 'recall': 0.992776886035313, 'f1-score': 0.9283302063789869, 'support': 1246}
2                    : {'precision': 0.7055971793741737, 'recall': 0.8724795640326976, 'f1-score': 0.780214424951267, 'support': 1835}
3                    : {'precision': 0.6578947368421053, 'recall': 0.7000811688311688, 'f1-score': 0.6783326779394416, 'support': 2464}
4                    : {'precision': 0.5707885304659498, 'recall': 0.6359400998336107, 'f1-score': 0.6016055406894381, 'support': 3005}
5                    : {'precision': 0.5002498750624688, 'recall': 0.556884561891516, 'f1-score': 0.527050151375543, 'support': 3595}
6                    : {'precision': 0.5897435897435898, 'recall': 0.5707468384633739, 'f1-score': 0.5800897295986419, 'support': 4191}
7                    : {'precision': 0.6143775838309601, 'recall': 0.5602094240837696, 'f1-score': 0.5860444736553839, 'support': 4775}
8                    : {'precision': 0.6099193072229876, 'recall': 0.5759152573871028, 'f1-score': 0.5924297457465111, 'support': 5381}
9                    : {'precision': 0.7566990291262136, 'recall': 0.6366606763600718, 'f1-score': 0.6915091828586638, 'support': 6121}
10                   : {'precision': 0.7170373767484521, 'recall': 0.5818756978042426, 'f1-score': 0.6424242424242423, 'support': 5374}
11                   : {'precision': 0.6030415285630727, 'recall': 0.6393137660190161, 'f1-score': 0.6206481388582322, 'support': 4838}
12                   : {'precision': 0.6441624365482234, 'recall': 0.6089251439539347, 'f1-score': 0.6260483473112974, 'support': 4168}
13                   : {'precision': 0.5716748060902039, 'recall': 0.5739832708393424, 'f1-score': 0.5728267127230858, 'support': 3467}
14                   : {'precision': 0.5475539349741719, 'recall': 0.6110545947778908, 'f1-score': 0.5775641025641025, 'support': 2949}
15                   : {'precision': 0.6178050652340752, 'recall': 0.6705539358600583, 'f1-score': 0.6430996604753345, 'support': 2401}
16                   : {'precision': 0.6475208640157094, 'recall': 0.7397644419517667, 'f1-score': 0.6905759162303665, 'support': 1783}
17                   : {'precision': 0.7209443099273608, 'recall': 0.9851116625310173, 'f1-score': 0.8325760223698008, 'support': 1209}
18                   : {'precision': 0.9783333333333334, 'recall': 0.9848993288590604, 'f1-score': 0.9816053511705685, 'support': 596}
accuracy             : 0.6398833333333334
macro avg            : {'precision': 0.6798304474579399, 'recall': 0.7099405616670748, 'f1-score': 0.6918246831305461, 'support': 60000}
weighted avg         : {'precision': 0.642944882879032, 'recall': 0.6398833333333334, 'f1-score': 0.638836088989473, 'support': 60000}
TEST : epoch=5 acc_l: 99.17 acc_o: 64.98 loss: 1.20823: 100%|██████████| 1875/1875 [00:26<00:00, 70.46it/s]


0                    : {'precision': 0.9954453441295547, 'recall': 0.996285666047611, 'f1-score': 0.9958653278204371, 'support': 5923}
1                    : {'precision': 0.9949592290585619, 'recall': 0.9954019578760012, 'f1-score': 0.9951805442277749, 'support': 6742}
2                    : {'precision': 0.9870194707938093, 'recall': 0.9954682779456193, 'f1-score': 0.9912258711456505, 'support': 5958}
3                    : {'precision': 0.9902676399026764, 'recall': 0.9957592562387865, 'f1-score': 0.9930058555627846, 'support': 6131}
4                    : {'precision': 0.9987749387469373, 'recall': 0.9768914755220814, 'f1-score': 0.9877120110764971, 'support': 5842}
5                    : {'precision': 0.9940564635958395, 'recall': 0.9872717210846708, 'f1-score': 0.9906524757056918, 'support': 5421}
6                    : {'precision': 0.9967796610169491, 'recall': 0.9937478877999324, 'f1-score': 0.9952614655610087, 'support': 5918}
7                    : {'precision': 0.9909365558912386, 'recall': 0.994732641660016, 'f1-score': 0.9928309702086984, 'support': 6265}
8                    : {'precision': 0.9832883187035787, 'recall': 0.9955563151598017, 'f1-score': 0.989384288747346, 'support': 5851}
9                    : {'precision': 0.9856950521709862, 'recall': 0.9845352160026896, 'f1-score': 0.9851147927003617, 'support': 5949}
accuracy             : 0.9917
macro avg            : {'precision': 0.9917222674010132, 'recall': 0.9915650415337209, 'f1-score': 0.991623360275625, 'support': 60000}
weighted avg         : {'precision': 0.99173222431324, 'recall': 0.9917, 'f1-score': 0.9916963267149583, 'support': 60000}
0                    : {'precision': 0.9931740614334471, 'recall': 0.9965753424657534, 'f1-score': 0.9948717948717949, 'support': 584}
1                    : {'precision': 0.9960505529225908, 'recall': 0.99213217938631, 'f1-score': 0.9940875049270793, 'support': 1271}
2                    : {'precision': 0.7581967213114754, 'recall': 0.9978425026968716, 'f1-score': 0.8616674429436424, 'support': 1854}
3                    : {'precision': 0.5998751560549314, 'recall': 0.7669592976855547, 'f1-score': 0.6732049036777584, 'support': 2506}
4                    : {'precision': 0.762107051826678, 'recall': 0.583984375, 'f1-score': 0.661260597124954, 'support': 3072}
5                    : {'precision': 0.623197362999588, 'recall': 0.8360972913211719, 'f1-score': 0.7141170915958451, 'support': 3618}
6                    : {'precision': 0.6777199232666484, 'recall': 0.5783442469597755, 'f1-score': 0.6241009463722398, 'support': 4276}
7                    : {'precision': 0.6230150546504434, 'recall': 0.633200586879061, 'f1-score': 0.6280665280665282, 'support': 4771}
8                    : {'precision': 0.7129977460555973, 'recall': 0.543424317617866, 'f1-score': 0.6167677642980937, 'support': 5239}
9                    : {'precision': 0.889618682722131, 'recall': 0.807819748177601, 'f1-score': 0.8467482851437007, 'support': 6036}
10                   : {'precision': 0.6262267696805178, 'recall': 0.5561943620178041, 'f1-score': 0.5891366270503879, 'support': 5392}
11                   : {'precision': 0.43918398768283295, 'recall': 0.4803199326457588, 'f1-score': 0.45883180858550315, 'support': 4751}
12                   : {'precision': 0.6563450373237489, 'recall': 0.5742622157716497, 'f1-score': 0.6125661205005806, 'support': 4134}
13                   : {'precision': 0.4920502092050209, 'recall': 0.3286752375628843, 'f1-score': 0.3941018766756032, 'support': 3578}
14                   : {'precision': 0.43944765814517633, 'recall': 0.8034800409416581, 'f1-score': 0.5681544028950543, 'support': 2931}
15                   : {'precision': 0.6720368239355581, 'recall': 0.49491525423728816, 'f1-score': 0.5700341630063446, 'support': 2360}
16                   : {'precision': 0.6662995594713657, 'recall': 0.6583242655059848, 'f1-score': 0.662287903667214, 'support': 1838}
17                   : {'precision': 0.6433370660694289, 'recall': 0.9939446366782007, 'f1-score': 0.7811012916383411, 'support': 1156}
18                   : {'precision': 0.9873617693522907, 'recall': 0.9873617693522907, 'f1-score': 0.9873617693522907, 'support': 633}
accuracy             : 0.6498166666666667
macro avg            : {'precision': 0.6978021681110247, 'recall': 0.7165188212054466, 'f1-score': 0.6967615169680502, 'support': 60000}
weighted avg         : {'precision': 0.6621802599553159, 'recall': 0.6498166666666667, 'f1-score': 0.6467713399522669, 'support': 60000}
TRAIN : epoch=6 acc_l: 98.93 acc_o: 66.23 loss: 1.20629: 100%|██████████| 1875/1875 [00:36<00:00, 51.35it/s]


TEST : epoch=6 acc_l: 0.27 acc_o: 0.18 loss: 0.00312:   0%|          | 0/1875 [00:00<?, ?it/s]0                    : {'precision': 0.9925738396624473, 'recall': 0.9929089988181665, 'f1-score': 0.9927413909520595, 'support': 5923}
1                    : {'precision': 0.9930329083901571, 'recall': 0.9936220706021952, 'f1-score': 0.9933274021352314, 'support': 6742}
2                    : {'precision': 0.9882609424786182, 'recall': 0.9890902987579725, 'f1-score': 0.9886754466907138, 'support': 5958}
3                    : {'precision': 0.9910145401078255, 'recall': 0.9893981405969663, 'f1-score': 0.990205680705191, 'support': 6131}
4                    : {'precision': 0.9910790873220107, 'recall': 0.9888736733995207, 'f1-score': 0.9899751520863679, 'support': 5842}
5                    : {'precision': 0.9861776631035754, 'recall': 0.9870872532743037, 'f1-score': 0.9866322485479857, 'support': 5421}
6                    : {'precision': 0.9915526271329617, 'recall': 0.9917201757350457, 'f1-score': 0.9916363943566783, 'support': 5918}
7                    : {'precision': 0.9880630272162979, 'recall': 0.990901835594573, 'f1-score': 0.9894803952821166, 'support': 6265}
8                    : {'precision': 0.9856312008210742, 'recall': 0.9847889249700906, 'f1-score': 0.9852098828759511, 'support': 5851}
9                    : {'precision': 0.9850218781555032, 'recall': 0.9838628340897629, 'f1-score': 0.9844420149693045, 'support': 5949}
accuracy             : 0.9893166666666666
macro avg            : {'precision': 0.9892407714390472, 'recall': 0.9892254205838598, 'f1-score': 0.98923260086016, 'support': 60000}
weighted avg         : {'precision': 0.9893168553871605, 'recall': 0.9893166666666666, 'f1-score': 0.9893162605243939, 'support': 60000}
0                    : {'precision': 0.9935379644588045, 'recall': 0.9903381642512077, 'f1-score': 0.9919354838709677, 'support': 621}
1                    : {'precision': 0.8373042886317222, 'recall': 0.9959514170040485, 'f1-score': 0.9097633136094674, 'support': 1235}
2                    : {'precision': 0.7120343839541547, 'recall': 0.8246681415929203, 'f1-score': 0.7642234751409532, 'support': 1808}
3                    : {'precision': 0.6832528778314148, 'recall': 0.7386591730228824, 'f1-score': 0.7098765432098766, 'support': 2491}
4                    : {'precision': 0.6305506216696269, 'recall': 0.6904376012965965, 'f1-score': 0.659136623858889, 'support': 3085}
5                    : {'precision': 0.5758825045091471, 'recall': 0.610822629133643, 'f1-score': 0.5928381962864722, 'support': 3659}
6                    : {'precision': 0.5797933409873708, 'recall': 0.5996200427451912, 'f1-score': 0.5895400420266168, 'support': 4211}
7                    : {'precision': 0.6421718273004797, 'recall': 0.6130308076602831, 'f1-score': 0.6272630457933972, 'support': 4804}
8                    : {'precision': 0.7102222222222222, 'recall': 0.6010908406996427, 'f1-score': 0.6511154120403383, 'support': 5317}
9                    : {'precision': 0.7332819425708229, 'recall': 0.6439329835843628, 'f1-score': 0.685709136781402, 'support': 5909}
10                   : {'precision': 0.6769920521703688, 'recall': 0.6083134956967589, 'f1-score': 0.6408179012345678, 'support': 5461}
11                   : {'precision': 0.619322278298486, 'recall': 0.5532417346500644, 'f1-score': 0.5844200022678309, 'support': 4658}
12                   : {'precision': 0.6269151138716356, 'recall': 0.7180460042684372, 'f1-score': 0.6693931690063003, 'support': 4217}
13                   : {'precision': 0.6230549380755795, 'recall': 0.5602512849800114, 'f1-score': 0.5899864682002707, 'support': 3502}
14                   : {'precision': 0.5753093872623001, 'recall': 0.6334330342306415, 'f1-score': 0.602973742486555, 'support': 3009}
15                   : {'precision': 0.6548259777538572, 'recall': 0.7439869547492866, 'f1-score': 0.6965648854961832, 'support': 2453}
16                   : {'precision': 0.7186351706036745, 'recall': 0.7626740947075209, 'f1-score': 0.7400000000000001, 'support': 1795}
17                   : {'precision': 0.7326007326007326, 'recall': 0.9852216748768473, 'f1-score': 0.8403361344537815, 'support': 1218}
18                   : {'precision': 0.9835164835164835, 'recall': 0.9817184643510055, 'f1-score': 0.9826166514181153, 'support': 547}
accuracy             : 0.6623
macro avg            : {'precision': 0.7004844267520465, 'recall': 0.7292336075527026, 'f1-score': 0.7120268540622099, 'support': 60000}
weighted avg         : {'precision': 0.6632275181307212, 'recall': 0.6623, 'f1-score': 0.6604166869786914, 'support': 60000}
TEST : epoch=6 acc_l: 99.25 acc_o: 69.23 loss: 1.16885: 100%|██████████| 1875/1875 [00:27<00:00, 69.17it/s]


0                    : {'precision': 0.9932761808707345, 'recall': 0.9976363329393888, 'f1-score': 0.9954514824797843, 'support': 5923}
1                    : {'precision': 0.9927675276752768, 'recall': 0.9976268169682587, 'f1-score': 0.9951912406599098, 'support': 6742}
2                    : {'precision': 0.9946353730092204, 'recall': 0.9958039610607586, 'f1-score': 0.9952193239956386, 'support': 5958}
3                    : {'precision': 0.9980308500164096, 'recall': 0.9920078290654053, 'f1-score': 0.9950102249488751, 'support': 6131}
4                    : {'precision': 0.9977547495682211, 'recall': 0.9888736733995207, 'f1-score': 0.9932943603851444, 'support': 5842}
5                    : {'precision': 0.99094771845557, 'recall': 0.9894853348090759, 'f1-score': 0.9902159867085103, 'support': 5421}
6                    : {'precision': 0.9751071546323772, 'recall': 0.9994930719837783, 'f1-score': 0.9871495327102804, 'support': 5918}
7                    : {'precision': 0.9947267497603068, 'recall': 0.9936153232242618, 'f1-score': 0.9941707258644096, 'support': 6265}
8                    : {'precision': 0.997907949790795, 'recall': 0.9782943086651854, 'f1-score': 0.9880037973591094, 'support': 5851}
9                    : {'precision': 0.9899227410144441, 'recall': 0.99075474869726, 'f1-score': 0.9903385701083761, 'support': 5949}
accuracy             : 0.9924833333333334
macro avg            : {'precision': 0.9925076994793356, 'recall': 0.9923591400812894, 'f1-score': 0.9924045245220038, 'support': 60000}
weighted avg         : {'precision': 0.9925441007055401, 'recall': 0.9924833333333334, 'f1-score': 0.9924852386357355, 'support': 60000}
0                    : {'precision': 0.988013698630137, 'recall': 0.9948275862068966, 'f1-score': 0.9914089347079038, 'support': 580}
1                    : {'precision': 0.9905882352941177, 'recall': 0.9976303317535545, 'f1-score': 0.9940968122786304, 'support': 1266}
2                    : {'precision': 0.7311301509587923, 'recall': 0.9983286908077994, 'f1-score': 0.8440885539331136, 'support': 1795}
3                    : {'precision': 0.7355182926829268, 'recall': 0.7457496136012365, 'f1-score': 0.740598618572525, 'support': 2588}
4                    : {'precision': 0.5652579967312631, 'recall': 0.7749679897567221, 'f1-score': 0.6537059538274604, 'support': 3124}
5                    : {'precision': 0.7546910755148741, 'recall': 0.46939937375462565, 'f1-score': 0.5787995787995788, 'support': 3513}
6                    : {'precision': 0.6641751201281367, 'recall': 0.8742094167252283, 'f1-score': 0.7548543689320389, 'support': 4269}
7                    : {'precision': 0.5608838740009403, 'recall': 0.49491806679112216, 'f1-score': 0.5258402203856749, 'support': 4821}
8                    : {'precision': 0.8264957264957264, 'recall': 0.5396205357142857, 'f1-score': 0.6529372045914922, 'support': 5376}
9                    : {'precision': 0.8675758908930928, 'recall': 0.6736122971818959, 'f1-score': 0.7583886164791847, 'support': 5855}
10                   : {'precision': 0.6541383989145183, 'recall': 0.8811917382562603, 'f1-score': 0.7508760999922124, 'support': 5471}
11                   : {'precision': 0.817841726618705, 'recall': 0.6102641185312433, 'f1-score': 0.6989670437776686, 'support': 4657}
12                   : {'precision': 0.7120660354454965, 'recall': 0.7134517149112138, 'f1-score': 0.7127582017010937, 'support': 4111}
13                   : {'precision': 0.5694711538461539, 'recall': 0.6632138857782754, 'f1-score': 0.6127780651836523, 'support': 3572}
14                   : {'precision': 0.4914011840992388, 'recall': 0.5835286240374958, 'f1-score': 0.5335169880624425, 'support': 2987}
15                   : {'precision': 0.67, 'recall': 0.5179007323026851, 'f1-score': 0.5842129417163836, 'support': 2458}
16                   : {'precision': 0.6759364358683314, 'recall': 0.6747875354107649, 'f1-score': 0.6753614970229658, 'support': 1765}
17                   : {'precision': 0.6753171856978085, 'recall': 0.9881856540084388, 'f1-score': 0.8023295649194929, 'support': 1185}
18                   : {'precision': 0.9950166112956811, 'recall': 0.9868204283360791, 'f1-score': 0.9909015715467329, 'support': 607}
accuracy             : 0.6922833333333334
macro avg            : {'precision': 0.7339746733218916, 'recall': 0.7464530702034644, 'f1-score': 0.7292853071805393, 'support': 60000}
weighted avg         : {'precision': 0.7097212430616681, 'recall': 0.6922833333333334, 'f1-score': 0.6884157229532526, 'support': 60000}
TRAIN : epoch=7 acc_l: 98.88 acc_o: 68.67 loss: 1.17976: 100%|██████████| 1875/1875 [00:36<00:00, 51.10it/s]


TEST : epoch=7 acc_l: 0.32 acc_o: 0.19 loss: 0.00405:   0%|          | 0/1875 [00:00<?, ?it/s]0                    : {'precision': 0.9920755353228797, 'recall': 0.9934154989025832, 'f1-score': 0.9927450649569766, 'support': 5923}
1                    : {'precision': 0.9931912374185908, 'recall': 0.9952536339365173, 'f1-score': 0.9942213661283152, 'support': 6742}
2                    : {'precision': 0.9862692565304756, 'recall': 0.9885867740852635, 'f1-score': 0.9874266554903603, 'support': 5958}
3                    : {'precision': 0.9910013089005235, 'recall': 0.9879301908334692, 'f1-score': 0.9894633668218574, 'support': 6131}
4                    : {'precision': 0.9903829641078482, 'recall': 0.9871619308456008, 'f1-score': 0.9887698242606087, 'support': 5842}
5                    : {'precision': 0.9850553505535056, 'recall': 0.9848736395498986, 'f1-score': 0.9849644866709714, 'support': 5421}
6                    : {'precision': 0.9900337837837838, 'recall': 0.9903683676917878, 'f1-score': 0.9902010474742355, 'support': 5918}
7                    : {'precision': 0.9887014640356461, 'recall': 0.9916999201915403, 'f1-score': 0.9901984221850346, 'support': 6265}
8                    : {'precision': 0.9856016455262255, 'recall': 0.9827379935053837, 'f1-score': 0.9841677364142062, 'support': 5851}
9                    : {'precision': 0.984196368527236, 'recall': 0.9840309295679946, 'f1-score': 0.9841136420946457, 'support': 5949}
accuracy             : 0.98875
macro avg            : {'precision': 0.9886508914706715, 'recall': 0.9886058879110038, 'f1-score': 0.988627161249721, 'support': 60000}
weighted avg         : {'precision': 0.988749272238301, 'recall': 0.98875, 'f1-score': 0.9887483929611633, 'support': 60000}
0                    : {'precision': 0.985, 'recall': 0.9966273187183811, 'f1-score': 0.9907795473595976, 'support': 593}
1                    : {'precision': 0.8854242204496011, 'recall': 0.9934906427990235, 'f1-score': 0.9363496932515338, 'support': 1229}
2                    : {'precision': 0.7275720164609053, 'recall': 0.9146404552509053, 'f1-score': 0.8104515241806096, 'support': 1933}
3                    : {'precision': 0.7088107467404188, 'recall': 0.7201926936973103, 'f1-score': 0.7144563918757467, 'support': 2491}
4                    : {'precision': 0.65744920993228, 'recall': 0.7429846938775511, 'f1-score': 0.6976047904191617, 'support': 3136}
5                    : {'precision': 0.6516788321167883, 'recall': 0.6267902274641954, 'f1-score': 0.6389922702547953, 'support': 3561}
6                    : {'precision': 0.6118825404051901, 'recall': 0.6335140230968654, 'f1-score': 0.6225104214914312, 'support': 4243}
7                    : {'precision': 0.6181586762834111, 'recall': 0.6051921079958463, 'f1-score': 0.6116066743624725, 'support': 4815}
8                    : {'precision': 0.7094109976092153, 'recall': 0.6078212290502794, 'f1-score': 0.6546986260154448, 'support': 5370}
9                    : {'precision': 0.7389918310420402, 'recall': 0.6243056724457162, 'f1-score': 0.6768248175182481, 'support': 5941}
10                   : {'precision': 0.6807228915662651, 'recall': 0.6946302657747243, 'f1-score': 0.6876062639821029, 'support': 5531}
11                   : {'precision': 0.7059970745977572, 'recall': 0.6219931271477663, 'f1-score': 0.6613382050696506, 'support': 4656}
12                   : {'precision': 0.6511185149928606, 'recall': 0.6681318681318681, 'f1-score': 0.6595154875256116, 'support': 4095}
13                   : {'precision': 0.6211145337440492, 'recall': 0.6395617070357554, 'f1-score': 0.6302031538570819, 'support': 3468}
14                   : {'precision': 0.6499007279947054, 'recall': 0.6632894292468761, 'f1-score': 0.6565268260070197, 'support': 2961}
15                   : {'precision': 0.6694884063305115, 'recall': 0.7678345293372731, 'f1-score': 0.7152968934329532, 'support': 2369}
16                   : {'precision': 0.743128435782109, 'recall': 0.7998924152770307, 'f1-score': 0.7704663212435233, 'support': 1859}
17                   : {'precision': 0.7529880478087649, 'recall': 0.9878048780487805, 'f1-score': 0.8545591559909571, 'support': 1148}
18                   : {'precision': 0.993322203672788, 'recall': 0.9900166389351082, 'f1-score': 0.9916666666666667, 'support': 601}
accuracy             : 0.6867
macro avg            : {'precision': 0.7243242056594559, 'recall': 0.7525638907016451, 'f1-score': 0.735865985816032, 'support': 60000}
weighted avg         : {'precision': 0.6870847808187306, 'recall': 0.6867, 'f1-score': 0.6847857276219432, 'support': 60000}
TEST : epoch=7 acc_l: 99.23 acc_o: 64.00 loss: 1.14380: 100%|██████████| 1875/1875 [00:27<00:00, 68.39it/s]


  0%|          | 0/1875 [00:00<?, ?it/s]0                    : {'precision': 0.9937856902922405, 'recall': 0.9989869998311667, 'f1-score': 0.9963795571272207, 'support': 5923}
1                    : {'precision': 0.9959899004901233, 'recall': 0.994660338178582, 'f1-score': 0.9953246753246754, 'support': 6742}
2                    : {'precision': 0.9858897742363878, 'recall': 0.9968110104061766, 'f1-score': 0.9913203138040394, 'support': 5958}
3                    : {'precision': 0.9845534995977474, 'recall': 0.9980427336486707, 'f1-score': 0.9912522274420864, 'support': 6131}
4                    : {'precision': 0.9891341256366724, 'recall': 0.9972612119137282, 'f1-score': 0.9931810433003752, 'support': 5842}
5                    : {'precision': 0.9966323666978485, 'recall': 0.9826600258254935, 'f1-score': 0.9895968790637192, 'support': 5421}
6                    : {'precision': 0.9974554707379135, 'recall': 0.9935789117945252, 'f1-score': 0.9955134174214849, 'support': 5918}
7                    : {'precision': 0.9888994608309547, 'recall': 0.9953711093375898, 'f1-score': 0.9921247315249384, 'support': 6265}
8                    : {'precision': 0.9939686369119421, 'recall': 0.985814390702444, 'f1-score': 0.9898747211257938, 'support': 5851}
9                    : {'precision': 0.9976006855184233, 'recall': 0.9784837787863506, 'f1-score': 0.9879497623896809, 'support': 5949}
accuracy             : 0.9923166666666666
macro avg            : {'precision': 0.9923909610950254, 'recall': 0.9921670510424727, 'f1-score': 0.9922517328524014, 'support': 60000}
weighted avg         : {'precision': 0.9923580754355299, 'recall': 0.9923166666666666, 'f1-score': 0.992310620457132, 'support': 60000}
0                    : {'precision': 0.9950166112956811, 'recall': 1.0, 'f1-score': 0.9975020815986678, 'support': 599}
1                    : {'precision': 0.9936808846761453, 'recall': 0.9976209357652657, 'f1-score': 0.9956470122675108, 'support': 1261}
2                    : {'precision': 0.7542236524537409, 'recall': 0.9978712080894092, 'f1-score': 0.8591065292096219, 'support': 1879}
3                    : {'precision': 0.6003220611916265, 'recall': 0.7564935064935064, 'f1-score': 0.669420003591309, 'support': 2464}
4                    : {'precision': 0.9865506329113924, 'recall': 0.411687025420931, 'f1-score': 0.5809457255998136, 'support': 3029}
5                    : {'precision': 0.6014809828340626, 'recall': 0.9944351697273233, 'f1-score': 0.7495805369127516, 'support': 3594}
6                    : {'precision': 0.5197516930022573, 'recall': 0.4336158192090395, 'f1-score': 0.47279260780287474, 'support': 4248}
7                    : {'precision': 0.5488203266787659, 'recall': 0.6398645789250952, 'f1-score': 0.5908558030480656, 'support': 4726}
8                    : {'precision': 0.564791034321737, 'recall': 0.44904399480230184, 'f1-score': 0.5003102378490175, 'support': 5387}
9                    : {'precision': 0.5723954556441866, 'recall': 0.3930942895086321, 'f1-score': 0.466095856707017, 'support': 6024}
10                   : {'precision': 0.5562097516099356, 'recall': 0.5538658849395383, 'f1-score': 0.5550353437987698, 'support': 5458}
11                   : {'precision': 0.6954424484364604, 'recall': 0.8703164029975021, 'f1-score': 0.7731139053254438, 'support': 4804}
12                   : {'precision': 0.9920769666100736, 'recall': 0.43028964162984784, 'f1-score': 0.6002396849854478, 'support': 4074}
13                   : {'precision': 0.504920838682071, 'recall': 0.6818838485986709, 'f1-score': 0.5802089735709896, 'support': 3461}
14                   : {'precision': 0.6154672395273899, 'recall': 0.5671395579016826, 'f1-score': 0.5903159340659341, 'support': 3031}
15                   : {'precision': 0.644754861681731, 'recall': 0.9890756302521009, 'f1-score': 0.7806333941303265, 'support': 2380}
16                   : {'precision': 0.9950940310711366, 'recall': 0.6694169416941694, 'f1-score': 0.8003946070371589, 'support': 1818}
17                   : {'precision': 0.6622093023255814, 'recall': 0.9793637145313844, 'f1-score': 0.7901491501907736, 'support': 1163}
18                   : {'precision': 1.0, 'recall': 0.9766666666666667, 'f1-score': 0.988195615514334, 'support': 600}
accuracy             : 0.6400333333333333
macro avg            : {'precision': 0.7264846723659988, 'recall': 0.7258813061659509, 'f1-score': 0.7021338422739909, 'support': 60000}
weighted avg         : {'precision': 0.6655476586908136, 'recall': 0.6400333333333333, 'f1-score': 0.628086318281494, 'support': 60000}
TRAIN : epoch=8 acc_l: 99.10 acc_o: 69.70 loss: 1.13818: 100%|██████████| 1875/1875 [00:37<00:00, 49.97it/s]


TEST : epoch=8 acc_l: 0.27 acc_o: 0.21 loss: 0.00268:   0%|          | 0/1875 [00:00<?, ?it/s]0                    : {'precision': 0.9952702702702703, 'recall': 0.994766165794361, 'f1-score': 0.995018154183906, 'support': 5923}
1                    : {'precision': 0.9943661971830986, 'recall': 0.9948086621180658, 'f1-score': 0.9945873804404242, 'support': 6742}
2                    : {'precision': 0.9907749077490775, 'recall': 0.9914400805639476, 'f1-score': 0.9911073825503356, 'support': 5958}
3                    : {'precision': 0.9929853181076672, 'recall': 0.9928233567117926, 'f1-score': 0.9929043308049914, 'support': 6131}
4                    : {'precision': 0.9912626349151962, 'recall': 0.9904142416980486, 'f1-score': 0.9908382567000599, 'support': 5842}
5                    : {'precision': 0.9902177925433739, 'recall': 0.9896698026194429, 'f1-score': 0.9899437217455485, 'support': 5421}
6                    : {'precision': 0.9932421017063693, 'recall': 0.993409935789118, 'f1-score': 0.9933260116583593, 'support': 5918}
7                    : {'precision': 0.9878980891719745, 'recall': 0.9902633679169992, 'f1-score': 0.9890793144679154, 'support': 6265}
8                    : {'precision': 0.9870218579234973, 'recall': 0.987865322167151, 'f1-score': 0.9874434099256856, 'support': 5851}
9                    : {'precision': 0.9868531939996629, 'recall': 0.9841990250462263, 'f1-score': 0.9855243225046287, 'support': 5949}
accuracy             : 0.9910333333333333
macro avg            : {'precision': 0.9909892363570189, 'recall': 0.9909659960425155, 'f1-score': 0.9909772284981855, 'support': 60000}
weighted avg         : {'precision': 0.9910333732138946, 'recall': 0.9910333333333333, 'f1-score': 0.9910329620496028, 'support': 60000}
0                    : {'precision': 0.9948453608247423, 'recall': 0.9948453608247423, 'f1-score': 0.9948453608247423, 'support': 582}
1                    : {'precision': 0.9944, 'recall': 0.9959935897435898, 'f1-score': 0.9951961569255404, 'support': 1248}
2                    : {'precision': 0.7474020783373302, 'recall': 0.9904661016949152, 'f1-score': 0.8519362186788155, 'support': 1888}
3                    : {'precision': 0.6846125863950527, 'recall': 0.749800796812749, 'f1-score': 0.7157254230842366, 'support': 2510}
4                    : {'precision': 0.6835978835978836, 'recall': 0.6585117227319062, 'f1-score': 0.6708203530633438, 'support': 2943}
5                    : {'precision': 0.646978397593656, 'recall': 0.6585026440300584, 'f1-score': 0.6526896551724137, 'support': 3593}
6                    : {'precision': 0.618603478699269, 'recall': 0.590045684058668, 'f1-score': 0.6039872015751907, 'support': 4159}
7                    : {'precision': 0.6125815808556925, 'recall': 0.6826262626262626, 'f1-score': 0.6457099178291611, 'support': 4950}
8                    : {'precision': 0.6671422747655932, 'recall': 0.61384096024006, 'f1-score': 0.6393826919320179, 'support': 5332}
9                    : {'precision': 0.8253211769581434, 'recall': 0.6656082887700535, 'f1-score': 0.7369102682701203, 'support': 5984}
10                   : {'precision': 0.7447733388532395, 'recall': 0.6700186219739293, 'f1-score': 0.7054210371532204, 'support': 5370}
11                   : {'precision': 0.6575716234652115, 'recall': 0.6106418918918919, 'f1-score': 0.6332384497481935, 'support': 4736}
12                   : {'precision': 0.6534813786722183, 'recall': 0.67680881648299, 'f1-score': 0.6649405672590327, 'support': 4174}
13                   : {'precision': 0.613618368962787, 'recall': 0.6501677852348994, 'f1-score': 0.6313645621181264, 'support': 3576}
14                   : {'precision': 0.6330097087378641, 'recall': 0.6691755046185426, 'f1-score': 0.6505903874937635, 'support': 2923}
15                   : {'precision': 0.7156537753222836, 'recall': 0.8045548654244307, 'f1-score': 0.757504873294347, 'support': 2415}
16                   : {'precision': 0.7685370741482966, 'recall': 0.8437843784378438, 'f1-score': 0.8044048243314106, 'support': 1818}
17                   : {'precision': 0.8103448275862069, 'recall': 0.9882253994953742, 'f1-score': 0.8904888215233042, 'support': 1189}
18                   : {'precision': 0.9805825242718447, 'recall': 0.9934426229508196, 'f1-score': 0.9869706840390878, 'support': 610}
accuracy             : 0.6970166666666666
macro avg            : {'precision': 0.7396346020024902, 'recall': 0.7635295420023013, 'f1-score': 0.7490593397008457, 'support': 60000}
weighted avg         : {'precision': 0.6993325113643679, 'recall': 0.6970166666666666, 'f1-score': 0.6957837831299839, 'support': 60000}
TEST : epoch=8 acc_l: 99.14 acc_o: 69.01 loss: 1.11556: 100%|██████████| 1875/1875 [00:27<00:00, 67.82it/s]


  0%|          | 0/1875 [00:00<?, ?it/s]0                    : {'precision': 0.9929530201342281, 'recall': 0.9991558331926389, 'f1-score': 0.9960447698392663, 'support': 5923}
1                    : {'precision': 0.9987993396368002, 'recall': 0.9870958172649066, 'f1-score': 0.9929130921298023, 'support': 6742}
2                    : {'precision': 0.9701760104302477, 'recall': 0.9991607922121517, 'f1-score': 0.984455101703324, 'support': 5958}
3                    : {'precision': 0.9967245332459875, 'recall': 0.9926602511825151, 'f1-score': 0.994688240581842, 'support': 6131}
4                    : {'precision': 0.9982590529247911, 'recall': 0.9815131804176652, 'f1-score': 0.9898152943207319, 'support': 5842}
5                    : {'precision': 0.9981371087928465, 'recall': 0.9883785279468733, 'f1-score': 0.9932338492909444, 'support': 5421}
6                    : {'precision': 0.9926025554808339, 'recall': 0.9976343359242987, 'f1-score': 0.9951120849485926, 'support': 5918}
7                    : {'precision': 0.9969150836174704, 'recall': 0.9800478850758181, 'f1-score': 0.9884095299420477, 'support': 6265}
8                    : {'precision': 0.9955418381344308, 'recall': 0.9923090070073491, 'f1-score': 0.9939227938029616, 'support': 5851}
9                    : {'precision': 0.9743589743589743, 'recall': 0.9964699949571356, 'f1-score': 0.9852904512590375, 'support': 5949}
accuracy             : 0.9913666666666666
macro avg            : {'precision': 0.9914467516756611, 'recall': 0.9914425625181351, 'f1-score': 0.991388520781855, 'support': 60000}
weighted avg         : {'precision': 0.9915065846516445, 'recall': 0.9913666666666666, 'f1-score': 0.991380436152909, 'support': 60000}
0                    : {'precision': 0.993127147766323, 'recall': 0.996551724137931, 'f1-score': 0.9948364888123924, 'support': 580}
1                    : {'precision': 0.9984338292873923, 'recall': 0.9922178988326849, 'f1-score': 0.9953161592505854, 'support': 1285}
2                    : {'precision': 0.9889006342494715, 'recall': 0.9946836788942052, 'f1-score': 0.9917837264776039, 'support': 1881}
3                    : {'precision': 0.7851550266207329, 'recall': 0.996422893481717, 'f1-score': 0.8782623927132599, 'support': 2516}
4                    : {'precision': 0.7916666666666666, 'recall': 0.7864139020537125, 'f1-score': 0.7890315422412426, 'support': 3165}
5                    : {'precision': 0.6991636798088411, 'recall': 0.81640625, 'f1-score': 0.753250096537521, 'support': 3584}
6                    : {'precision': 0.5710391822827938, 'recall': 0.41733067729083667, 'f1-score': 0.4822327722629837, 'support': 4016}
7                    : {'precision': 0.604674625861779, 'recall': 0.7367342757631633, 'f1-score': 0.6642039157739196, 'support': 4881}
8                    : {'precision': 0.9951554299555915, 'recall': 0.44826332060374613, 'f1-score': 0.6181043129388164, 'support': 5499}
9                    : {'precision': 0.6891017344033135, 'recall': 0.8938885157824042, 'f1-score': 0.7782487940359596, 'support': 5956}
10                   : {'precision': 0.5937888198757764, 'recall': 0.44440312383785796, 'f1-score': 0.5083483994469851, 'support': 5378}
11                   : {'precision': 0.5979924304755636, 'recall': 0.7689377909437156, 'f1-score': 0.672776080718319, 'support': 4726}
12                   : {'precision': 0.7624053826745164, 'recall': 0.43518963034085456, 'f1-score': 0.5540953545232274, 'support': 4166}
13                   : {'precision': 0.49978614200171084, 'recall': 0.6791630340017437, 'f1-score': 0.5758285080694838, 'support': 3441}
14                   : {'precision': 0.7610891523935002, 'recall': 0.5914675767918088, 'f1-score': 0.6656424044555407, 'support': 2930}
15                   : {'precision': 0.5779313632030505, 'recall': 0.9893920848633211, 'f1-score': 0.729652474800662, 'support': 2451}
16                   : {'precision': 0.9969183359013868, 'recall': 0.35568993952721273, 'f1-score': 0.5243111831442463, 'support': 1819}
17                   : {'precision': 0.6426953567383918, 'recall': 0.9947414548641542, 'f1-score': 0.780873753009976, 'support': 1141}
18                   : {'precision': 0.9749163879598662, 'recall': 0.9965811965811966, 'f1-score': 0.9856297548605241, 'support': 585}
accuracy             : 0.6900666666666667
macro avg            : {'precision': 0.7644179646382456, 'recall': 0.7544462615048563, 'f1-score': 0.7338120060038552, 'support': 60000}
weighted avg         : {'precision': 0.7201058532896382, 'recall': 0.6900666666666667, 'f1-score': 0.6784038664992059, 'support': 60000}
TRAIN : epoch=9 acc_l: 99.13 acc_o: 73.28 loss: 1.11227: 100%|██████████| 1875/1875 [00:38<00:00, 49.21it/s]


TEST : epoch=9 acc_l: 0.32 acc_o: 0.24 loss: 0.00327:   0%|          | 0/1875 [00:00<?, ?it/s]0                    : {'precision': 0.9927572848239852, 'recall': 0.9951038325173054, 'f1-score': 0.993929173693086, 'support': 5923}
1                    : {'precision': 0.9934805156319455, 'recall': 0.9945120142390982, 'f1-score': 0.9939959973315543, 'support': 6742}
2                    : {'precision': 0.989256337082424, 'recall': 0.9890902987579725, 'f1-score': 0.9891733109525808, 'support': 5958}
3                    : {'precision': 0.9931551499348109, 'recall': 0.9939650954167346, 'f1-score': 0.9935599576098475, 'support': 6131}
4                    : {'precision': 0.9905757368060315, 'recall': 0.9895583704210886, 'f1-score': 0.9900667922589484, 'support': 5842}
5                    : {'precision': 0.9916958848496032, 'recall': 0.9913300129127467, 'f1-score': 0.9915129151291513, 'support': 5421}
6                    : {'precision': 0.993573482157957, 'recall': 0.992734031767489, 'f1-score': 0.9931535795790719, 'support': 5918}
7                    : {'precision': 0.9910571702331523, 'recall': 0.9905826017557862, 'f1-score': 0.9908198291689949, 'support': 6265}
8                    : {'precision': 0.9912731006160165, 'recall': 0.99008716458725, 'f1-score': 0.9906797776827705, 'support': 5851}
9                    : {'precision': 0.986211535227846, 'recall': 0.9858799798285426, 'f1-score': 0.9860457296570276, 'support': 5949}
accuracy             : 0.9913333333333333
macro avg            : {'precision': 0.9913036197363774, 'recall': 0.9912843402204012, 'f1-score': 0.9912937063063032, 'support': 60000}
weighted avg         : {'precision': 0.9913324955812873, 'recall': 0.9913333333333333, 'f1-score': 0.991332640808185, 'support': 60000}
0                    : {'precision': 0.9948542024013722, 'recall': 0.9931506849315068, 'f1-score': 0.9940017137960583, 'support': 584}
1                    : {'precision': 0.9957947855340622, 'recall': 0.9966329966329966, 'f1-score': 0.9962137147665124, 'support': 1188}
2                    : {'precision': 0.8493333333333334, 'recall': 0.9927272727272727, 'f1-score': 0.915449101796407, 'support': 1925}
3                    : {'precision': 0.7397540983606558, 'recall': 0.852755905511811, 'f1-score': 0.7922457937088515, 'support': 2540}
4                    : {'precision': 0.7053805774278216, 'recall': 0.7093368525239195, 'f1-score': 0.7073531830893239, 'support': 3031}
5                    : {'precision': 0.6315093840867037, 'recall': 0.673716864072194, 'f1-score': 0.6519306863146405, 'support': 3546}
6                    : {'precision': 0.6406787434074753, 'recall': 0.6589622641509434, 'f1-score': 0.649691896291129, 'support': 4240}
7                    : {'precision': 0.7202937249666221, 'recall': 0.6634556261529002, 'f1-score': 0.6907073509015257, 'support': 4879}
8                    : {'precision': 0.7436858721389108, 'recall': 0.6984803558191253, 'f1-score': 0.720374617737003, 'support': 5396}
9                    : {'precision': 0.8174118573961, 'recall': 0.6900565347522447, 'f1-score': 0.7483545216842485, 'support': 6014}
10                   : {'precision': 0.7559188275084555, 'recall': 0.7430735131141485, 'f1-score': 0.749441132637854, 'support': 5414}
11                   : {'precision': 0.7151653363740023, 'recall': 0.6748439853669034, 'f1-score': 0.6944198405668733, 'support': 4647}
12                   : {'precision': 0.6708624708624709, 'recall': 0.687529861442905, 'f1-score': 0.6790939122227465, 'support': 4186}
13                   : {'precision': 0.6599397590361445, 'recall': 0.6363636363636364, 'f1-score': 0.6479373059293213, 'support': 3443}
14                   : {'precision': 0.681238034460753, 'recall': 0.720310391363023, 'f1-score': 0.7002295834699902, 'support': 2964}
15                   : {'precision': 0.7063011747953009, 'recall': 0.8091353996737357, 'f1-score': 0.7542292339859343, 'support': 2452}
16                   : {'precision': 0.772484200291687, 'recall': 0.8692560175054704, 'f1-score': 0.8180180180180181, 'support': 1828}
17                   : {'precision': 0.8549734244495064, 'recall': 0.990325417766051, 'f1-score': 0.9176854115729421, 'support': 1137}
18                   : {'precision': 0.9880341880341881, 'recall': 0.9863481228668942, 'f1-score': 0.9871904355251921, 'support': 586}
accuracy             : 0.7328333333333333
macro avg            : {'precision': 0.7707165260455561, 'recall': 0.7919190369861938, 'f1-score': 0.7797140765270827, 'support': 60000}
weighted avg         : {'precision': 0.7337578578164822, 'recall': 0.7328333333333333, 'f1-score': 0.7317790790496795, 'support': 60000}
TEST : epoch=9 acc_l: 99.20 acc_o: 73.61 loss: 1.09934: 100%|██████████| 1875/1875 [00:28<00:00, 66.57it/s]


0                    : {'precision': 0.998465996250213, 'recall': 0.9890258315043052, 'f1-score': 0.9937234944868533, 'support': 5923}
1                    : {'precision': 0.9833673767143274, 'recall': 0.9997033521210323, 'f1-score': 0.9914680788467197, 'support': 6742}
2                    : {'precision': 0.9886892880904857, 'recall': 0.9976502181940249, 'f1-score': 0.9931495405179617, 'support': 5958}
3                    : {'precision': 0.9991747813170491, 'recall': 0.987440874245637, 'f1-score': 0.9932731747333882, 'support': 6131}
4                    : {'precision': 0.9967269595176572, 'recall': 0.9904142416980486, 'f1-score': 0.9935605735382502, 'support': 5842}
5                    : {'precision': 0.9864815491413957, 'recall': 0.9961261759822911, 'f1-score': 0.9912804038549794, 'support': 5421}
6                    : {'precision': 0.9910999160369437, 'recall': 0.9972963839134843, 'f1-score': 0.994188494904405, 'support': 5918}
7                    : {'precision': 0.9966194462330973, 'recall': 0.9881883479648843, 'f1-score': 0.9923859902220085, 'support': 6265}
8                    : {'precision': 0.9831023994592768, 'recall': 0.9943599384720561, 'f1-score': 0.9886991248194409, 'support': 5851}
9                    : {'precision': 0.9972621492128679, 'recall': 0.979660447133972, 'f1-score': 0.9883829390316289, 'support': 5949}
accuracy             : 0.9920166666666667
macro avg            : {'precision': 0.9920989861973315, 'recall': 0.9919865811229736, 'f1-score': 0.9920111814955636, 'support': 60000}
weighted avg         : {'precision': 0.9920819747725697, 'recall': 0.9920166666666667, 'f1-score': 0.9920171867872967, 'support': 60000}
0                    : {'precision': 0.9982547993019197, 'recall': 0.9828178694158075, 'f1-score': 0.9904761904761905, 'support': 582}
1                    : {'precision': 0.9917627677100495, 'recall': 0.9958643507030603, 'f1-score': 0.9938093272802311, 'support': 1209}
2                    : {'precision': 0.752442996742671, 'recall': 0.9956896551724138, 'f1-score': 0.8571428571428572, 'support': 1856}
3                    : {'precision': 0.7519068647129666, 'recall': 0.7607636068237206, 'f1-score': 0.7563093074904099, 'support': 2462}
4                    : {'precision': 0.7866149369544132, 'recall': 0.7953579601176856, 'f1-score': 0.790962288686606, 'support': 3059}
5                    : {'precision': 0.6669474866610503, 'recall': 0.6549917264202979, 'f1-score': 0.6609155419507445, 'support': 3626}
6                    : {'precision': 0.6331877729257642, 'recall': 0.7196880170172536, 'f1-score': 0.6736725663716815, 'support': 4231}
7                    : {'precision': 0.7340511986997156, 'recall': 0.7561741314357472, 'f1-score': 0.7449484536082475, 'support': 4778}
8                    : {'precision': 0.8497067448680352, 'recall': 0.6467633928571429, 'f1-score': 0.7344740177439797, 'support': 5376}
9                    : {'precision': 0.786161449752883, 'recall': 0.7993299832495813, 'f1-score': 0.7926910299003322, 'support': 5970}
10                   : {'precision': 0.8560551986676184, 'recall': 0.6662962962962963, 'f1-score': 0.7493491617202958, 'support': 5400}
11                   : {'precision': 0.5706102117061022, 'recall': 0.4840481724065075, 'f1-score': 0.5237768632830362, 'support': 4733}
12                   : {'precision': 0.5516731373636532, 'recall': 0.7154159673939103, 'f1-score': 0.6229645093945722, 'support': 4171}
13                   : {'precision': 0.8049349783261087, 'recall': 0.678280415847148, 'f1-score': 0.7362000609942057, 'support': 3559}
14                   : {'precision': 0.663475485929794, 'recall': 0.7768342391304348, 'f1-score': 0.7156939446096072, 'support': 2944}
15                   : {'precision': 0.7413862991487636, 'recall': 0.7456176110884631, 'f1-score': 0.7434959349593496, 'support': 2453}
16                   : {'precision': 0.7450248756218906, 'recall': 0.992817679558011, 'f1-score': 0.851255329227854, 'support': 1810}
17                   : {'precision': 0.9897348160821214, 'recall': 0.985519591141397, 'f1-score': 0.987622705932565, 'support': 1174}
18                   : {'precision': 0.9983277591973244, 'recall': 0.9835255354200988, 'f1-score': 0.9908713692946058, 'support': 607}
accuracy             : 0.7361
macro avg            : {'precision': 0.7827505147564655, 'recall': 0.796620852710262, 'f1-score': 0.7850858663193353, 'support': 60000}
weighted avg         : {'precision': 0.7446265331812568, 'recall': 0.7361, 'f1-score': 0.7352334626564165, 'support': 60000}
TRAIN : epoch=10 acc_l: 99.56 acc_o: 81.00 loss: 1.06512: 100%|██████████| 1875/1875 [00:38<00:00, 48.74it/s]


TEST : epoch=10 acc_l: 0.59 acc_o: 0.48 loss: 0.00601:   0%|          | 7/1875 [00:00<00:27, 67.86it/s]0                    : {'precision': 0.9972991222147198, 'recall': 0.9974674995779166, 'f1-score': 0.9973833037899891, 'support': 5923}
1                    : {'precision': 0.9958549222797928, 'recall': 0.9977751409077426, 'f1-score': 0.9968141068385568, 'support': 6742}
2                    : {'precision': 0.9963087248322148, 'recall': 0.9966431688486069, 'f1-score': 0.9964759187783185, 'support': 5958}
3                    : {'precision': 0.9977112963871179, 'recall': 0.9954330451802316, 'f1-score': 0.9965708687132592, 'support': 6131}
4                    : {'precision': 0.9960602946214457, 'recall': 0.9953782951044163, 'f1-score': 0.9957191780821919, 'support': 5842}
5                    : {'precision': 0.994103556292611, 'recall': 0.9952038369304557, 'f1-score': 0.9946533923303835, 'support': 5421}
6                    : {'precision': 0.9971264367816092, 'recall': 0.9967894558972626, 'f1-score': 0.9969579178637824, 'support': 5918}
7                    : {'precision': 0.9948971455908149, 'recall': 0.9958499600957702, 'f1-score': 0.9953733248245055, 'support': 6265}
8                    : {'precision': 0.994016071123269, 'recall': 0.9936762946504871, 'f1-score': 0.9938461538461538, 'support': 5851}
9                    : {'precision': 0.9925963318189467, 'recall': 0.9915952260884182, 'f1-score': 0.9920955264043054, 'support': 5949}
accuracy             : 0.9956166666666667
macro avg            : {'precision': 0.9955973901942542, 'recall': 0.9955811923281308, 'f1-score': 0.9955889691471447, 'support': 60000}
weighted avg         : {'precision': 0.9956170004593635, 'recall': 0.9956166666666667, 'f1-score': 0.9956164997611916, 'support': 60000}
0                    : {'precision': 1.0, 'recall': 0.9983079526226735, 'f1-score': 0.9991532599491956, 'support': 591}
1                    : {'precision': 0.9984627209838586, 'recall': 0.9976958525345622, 'f1-score': 0.9980791394544757, 'support': 1302}
2                    : {'precision': 0.7774086378737541, 'recall': 0.9973361747469366, 'f1-score': 0.8737456242707117, 'support': 1877}
3                    : {'precision': 0.7763950220794862, 'recall': 0.7849025974025974, 'f1-score': 0.7806256306760848, 'support': 2464}
4                    : {'precision': 0.7815656565656566, 'recall': 0.8160843770599868, 'f1-score': 0.798452112221864, 'support': 3034}
5                    : {'precision': 0.7517388978063135, 'recall': 0.7915492957746478, 'f1-score': 0.7711306256860593, 'support': 3550}
6                    : {'precision': 0.7472578763127188, 'recall': 0.7758662466682821, 'f1-score': 0.7612933903946744, 'support': 4127}
7                    : {'precision': 0.7878038730943552, 'recall': 0.7863458770306395, 'f1-score': 0.7870741998559225, 'support': 4863}
8                    : {'precision': 0.9281705948372615, 'recall': 0.7665925101965146, 'f1-score': 0.8396791552441871, 'support': 5394}
9                    : {'precision': 0.8883534136546185, 'recall': 0.8990408063729475, 'f1-score': 0.8936651583710408, 'support': 6151}
10                   : {'precision': 0.9214301981325438, 'recall': 0.7545691906005222, 'f1-score': 0.8296934276632831, 'support': 5362}
11                   : {'precision': 0.7622632345798932, 'recall': 0.6657476139978791, 'f1-score': 0.7107438016528926, 'support': 4715}
12                   : {'precision': 0.6944611077784443, 'recall': 0.8334533237341013, 'f1-score': 0.7576352530541012, 'support': 4167}
13                   : {'precision': 0.8223644119439366, 'recall': 0.7518105849582173, 'f1-score': 0.7855064027939465, 'support': 3590}
14                   : {'precision': 0.7133182844243793, 'recall': 0.7688564476885644, 'f1-score': 0.7400468384074941, 'support': 2877}
15                   : {'precision': 0.7246677740863787, 'recall': 0.7606800348735833, 'f1-score': 0.7422373458102934, 'support': 2294}
16                   : {'precision': 0.7603448275862069, 'recall': 0.9932432432432432, 'f1-score': 0.861328125, 'support': 1776}
17                   : {'precision': 0.9968578161822467, 'recall': 0.9937353171495693, 'f1-score': 0.9952941176470589, 'support': 1277}
18                   : {'precision': 0.988155668358714, 'recall': 0.9915110356536503, 'f1-score': 0.9898305084745763, 'support': 589}
accuracy             : 0.81005
macro avg            : {'precision': 0.8326852640147774, 'recall': 0.8488067622267959, 'f1-score': 0.8376428482435717, 'support': 60000}
weighted avg         : {'precision': 0.8164981381029216, 'recall': 0.81005, 'f1-score': 0.8100265924623091, 'support': 60000}
TEST : epoch=10 acc_l: 99.73 acc_o: 81.61 loss: 1.04828: 100%|██████████| 1875/1875 [00:28<00:00, 66.45it/s]


0                    : {'precision': 0.9974730458221024, 'recall': 0.9996623332770556, 'f1-score': 0.9985664895859685, 'support': 5923}
1                    : {'precision': 0.9979225404362665, 'recall': 0.9974784930287749, 'f1-score': 0.9977004673243826, 'support': 6742}
2                    : {'precision': 0.9984868863483524, 'recall': 0.9968110104061766, 'f1-score': 0.9976482445825634, 'support': 5958}
3                    : {'precision': 0.9985306122448979, 'recall': 0.9975534170608383, 'f1-score': 0.9980417754569191, 'support': 6131}
4                    : {'precision': 0.9979423868312757, 'recall': 0.9962341663813763, 'f1-score': 0.997087544971732, 'support': 5842}
5                    : {'precision': 0.9977839335180055, 'recall': 0.9966795794133924, 'f1-score': 0.9972314507198228, 'support': 5421}
6                    : {'precision': 0.9976367319378798, 'recall': 0.9986481919567421, 'f1-score': 0.9981422057084952, 'support': 5918}
7                    : {'precision': 0.995379958578939, 'recall': 0.9972865123703113, 'f1-score': 0.9963323233933982, 'support': 6265}
8                    : {'precision': 0.9972635539592953, 'recall': 0.9965817808921552, 'f1-score': 0.9969225508633955, 'support': 5851}
9                    : {'precision': 0.9947960382742992, 'recall': 0.9961338040006724, 'f1-score': 0.9954644716949438, 'support': 5949}
accuracy             : 0.9973166666666666
macro avg            : {'precision': 0.9973215687951311, 'recall': 0.9973069288787496, 'f1-score': 0.9973137524301622, 'support': 60000}
weighted avg         : {'precision': 0.9973178184007845, 'recall': 0.9973166666666666, 'f1-score': 0.9973167488967074, 'support': 60000}
0                    : {'precision': 0.9948630136986302, 'recall': 0.9982817869415808, 'f1-score': 0.9965694682675814, 'support': 582}
1                    : {'precision': 1.0, 'recall': 0.9992025518341308, 'f1-score': 0.9996011168727563, 'support': 1254}
2                    : {'precision': 0.9979068550497122, 'recall': 0.9984293193717277, 'f1-score': 0.9981680188432348, 'support': 1910}
3                    : {'precision': 0.8030656447850717, 'recall': 0.9975165562913907, 'f1-score': 0.8897913974524644, 'support': 2416}
4                    : {'precision': 0.6571732765504392, 'recall': 0.8058093994778068, 'f1-score': 0.7239407711479255, 'support': 3064}
5                    : {'precision': 0.8014378637452927, 'recall': 0.6434854315557998, 'f1-score': 0.7138283274889464, 'support': 3638}
6                    : {'precision': 0.7505678298575263, 'recall': 0.862805601708996, 'f1-score': 0.8027826855123675, 'support': 4213}
7                    : {'precision': 0.7261691542288558, 'recall': 0.7523711340206185, 'f1-score': 0.7390379746835443, 'support': 4850}
8                    : {'precision': 0.9997519225998511, 'recall': 0.7446415373244641, 'f1-score': 0.8535423064704014, 'support': 5412}
9                    : {'precision': 0.9153810191678354, 'recall': 0.9969450101832994, 'f1-score': 0.9544235924932977, 'support': 5892}
10                   : {'precision': 0.8712384851586489, 'recall': 0.7900501206608502, 'f1-score': 0.8286604361370717, 'support': 5387}
11                   : {'precision': 0.8330560177481975, 'recall': 0.6216887417218543, 'f1-score': 0.7120170656553685, 'support': 4832}
12                   : {'precision': 0.6997781612028593, 'recall': 0.696174595389897, 'f1-score': 0.6979717271051015, 'support': 4078}
13                   : {'precision': 0.7068396226415095, 'recall': 0.8473282442748091, 'f1-score': 0.7707342162787708, 'support': 3537}
14                   : {'precision': 0.8106880894166958, 'recall': 0.792420621372482, 'f1-score': 0.8014502762430938, 'support': 2929}
15                   : {'precision': 0.7470978441127695, 'recall': 0.7527151211361738, 'f1-score': 0.7498959633791094, 'support': 2394}
16                   : {'precision': 0.7556203164029975, 'recall': 0.9967051070840197, 'f1-score': 0.8595784986976083, 'support': 1821}
17                   : {'precision': 0.9958779884583677, 'recall': 0.9991728701406121, 'f1-score': 0.9975227085053674, 'support': 1209}
18                   : {'precision': 0.998272884283247, 'recall': 0.993127147766323, 'f1-score': 0.9956933677863911, 'support': 582}
accuracy             : 0.81615
macro avg            : {'precision': 0.8455150520583423, 'recall': 0.8573089946450968, 'f1-score': 0.8465899957379159, 'support': 60000}
weighted avg         : {'precision': 0.8248850484562459, 'recall': 0.81615, 'f1-score': 0.8147410022572239, 'support': 60000}
TRAIN : epoch=11 acc_l: 99.67 acc_o: 82.61 loss: 1.05070: 100%|██████████| 1875/1875 [00:36<00:00, 50.80it/s]


TEST : epoch=11 acc_l: 0.32 acc_o: 0.26 loss: 0.00358:   0%|          | 0/1875 [00:00<?, ?it/s]0                    : {'precision': 0.9978066475451325, 'recall': 0.9984804997467499, 'f1-score': 0.9981434599156118, 'support': 5923}
1                    : {'precision': 0.9979228486646884, 'recall': 0.9976268169682587, 'f1-score': 0.9977748108589231, 'support': 6742}
2                    : {'precision': 0.9979838709677419, 'recall': 0.9969788519637462, 'f1-score': 0.9974811083123425, 'support': 5958}
3                    : {'precision': 0.9980430528375733, 'recall': 0.9982058391779481, 'f1-score': 0.9981244393704639, 'support': 6131}
4                    : {'precision': 0.9970855477455854, 'recall': 0.9955494693598083, 'f1-score': 0.9963169164882226, 'support': 5842}
5                    : {'precision': 0.9955768521931441, 'recall': 0.9964951116030253, 'f1-score': 0.9960357702590579, 'support': 5421}
6                    : {'precision': 0.9969620253164557, 'recall': 0.9981412639405205, 'f1-score': 0.9975512961242929, 'support': 5918}
7                    : {'precision': 0.9955364259524948, 'recall': 0.9968076616121309, 'f1-score': 0.9961716382198117, 'support': 6265}
8                    : {'precision': 0.9962354551676934, 'recall': 0.995043582293625, 'f1-score': 0.9956391620350576, 'support': 5851}
9                    : {'precision': 0.9939475453934096, 'recall': 0.9937804673054295, 'f1-score': 0.9938639993275616, 'support': 5949}
accuracy             : 0.9967333333333334
macro avg            : {'precision': 0.9967100271783919, 'recall': 0.9967109563971241, 'f1-score': 0.9967102600911346, 'support': 60000}
weighted avg         : {'precision': 0.9967335822515933, 'recall': 0.9967333333333334, 'f1-score': 0.996733229311205, 'support': 60000}
0                    : {'precision': 1.0, 'recall': 1.0, 'f1-score': 1.0, 'support': 588}
1                    : {'precision': 0.9977272727272727, 'recall': 0.9992412746585736, 'f1-score': 0.9984836997725549, 'support': 1318}
2                    : {'precision': 0.8008752735229759, 'recall': 0.997275204359673, 'f1-score': 0.8883495145631067, 'support': 1835}
3                    : {'precision': 0.8024340770791075, 'recall': 0.8119868637110016, 'f1-score': 0.8071822077127117, 'support': 2436}
4                    : {'precision': 0.7339233038348083, 'recall': 0.8357406785354383, 'f1-score': 0.7815297628396418, 'support': 2977}
5                    : {'precision': 0.796029919447641, 'recall': 0.755598033861278, 'f1-score': 0.7752871952927991, 'support': 3662}
6                    : {'precision': 0.6818268679533728, 'recall': 0.8334501284746555, 'f1-score': 0.7500525541307548, 'support': 4281}
7                    : {'precision': 0.8874533448176859, 'recall': 0.6488245172124265, 'f1-score': 0.7496059173032619, 'support': 4764}
8                    : {'precision': 0.9331692540704278, 'recall': 0.9044036697247706, 'f1-score': 0.9185613119642192, 'support': 5450}
9                    : {'precision': 0.942141769582135, 'recall': 0.9386482386650935, 'f1-score': 0.9403917595406958, 'support': 5933}
10                   : {'precision': 0.9873959571938169, 'recall': 0.7724651162790698, 'f1-score': 0.8668058455114823, 'support': 5375}
11                   : {'precision': 0.7775368603642672, 'recall': 0.7544708605091521, 'f1-score': 0.7658302189001601, 'support': 4753}
12                   : {'precision': 0.7285438765670202, 'recall': 0.7260932244113407, 'f1-score': 0.7273164861612516, 'support': 4162}
13                   : {'precision': 0.6858006042296072, 'recall': 0.7138364779874213, 'f1-score': 0.6995377503852079, 'support': 3498}
14                   : {'precision': 0.7019316493313521, 'recall': 0.7897024406552993, 'f1-score': 0.7432347388294526, 'support': 2991}
15                   : {'precision': 0.7828902522154055, 'recall': 0.9441019317714755, 'f1-score': 0.8559716787777156, 'support': 2433}
16                   : {'precision': 0.9340546110252447, 'recall': 0.9939692982456141, 'f1-score': 0.9630810092961488, 'support': 1824}
17                   : {'precision': 0.9929639401934917, 'recall': 0.9955908289241623, 'f1-score': 0.9942756494936151, 'support': 1134}
18                   : {'precision': 0.9948630136986302, 'recall': 0.9914675767918089, 'f1-score': 0.9931623931623933, 'support': 586}
accuracy             : 0.8260666666666666
macro avg            : {'precision': 0.850608518308119, 'recall': 0.8635192823567504, 'f1-score': 0.8536136680861669, 'support': 60000}
weighted avg         : {'precision': 0.8350085253550688, 'recall': 0.8260666666666666, 'f1-score': 0.8263106334160236, 'support': 60000}
TEST : epoch=11 acc_l: 99.76 acc_o: 81.07 loss: 1.04138: 100%|██████████| 1875/1875 [00:27<00:00, 69.12it/s]


0                    : {'precision': 0.9986504723346828, 'recall': 0.9994934999155833, 'f1-score': 0.9990718082862206, 'support': 5923}
1                    : {'precision': 0.9983681946298769, 'recall': 0.998220112726194, 'f1-score': 0.9982941481866054, 'support': 6742}
2                    : {'precision': 0.9984884111521666, 'recall': 0.9978180597515945, 'f1-score': 0.998153122901276, 'support': 5958}
3                    : {'precision': 0.9986928104575163, 'recall': 0.9969009949437286, 'f1-score': 0.9977960982776916, 'support': 6131}
4                    : {'precision': 0.9977739726027397, 'recall': 0.9974323861691201, 'f1-score': 0.9976031501455229, 'support': 5842}
5                    : {'precision': 0.998520162782094, 'recall': 0.9957572403615569, 'f1-score': 0.9971367876604784, 'support': 5421}
6                    : {'precision': 0.9971327373924777, 'recall': 0.9989861439675566, 'f1-score': 0.998058580231282, 'support': 5918}
7                    : {'precision': 0.9964906683681608, 'recall': 0.9971268954509178, 'f1-score': 0.9968086803893411, 'support': 6265}
8                    : {'precision': 0.9965846994535519, 'recall': 0.9974363356691164, 'f1-score': 0.9970103356965918, 'support': 5851}
9                    : {'precision': 0.9949639080073863, 'recall': 0.996301899478904, 'f1-score': 0.9956324542247607, 'support': 5949}
accuracy             : 0.9975666666666667
macro avg            : {'precision': 0.9975666037180654, 'recall': 0.9975473568434273, 'f1-score': 0.9975565165999771, 'support': 60000}
weighted avg         : {'precision': 0.9975676830538902, 'recall': 0.9975666666666667, 'f1-score': 0.9975667297698186, 'support': 60000}
0                    : {'precision': 0.9949664429530202, 'recall': 1.0, 'f1-score': 0.9974768713204373, 'support': 593}
1                    : {'precision': 1.0, 'recall': 1.0, 'f1-score': 1.0, 'support': 1268}
2                    : {'precision': 0.7597299444003177, 'recall': 0.9994775339602926, 'f1-score': 0.8632671480144405, 'support': 1914}
3                    : {'precision': 0.7432216905901117, 'recall': 0.7540453074433657, 'f1-score': 0.7485943775100401, 'support': 2472}
4                    : {'precision': 0.7971745711402624, 'recall': 0.7860696517412935, 'f1-score': 0.7915831663326652, 'support': 3015}
5                    : {'precision': 0.7112511671335201, 'recall': 0.8352521929824561, 'f1-score': 0.7682803832576904, 'support': 3648}
6                    : {'precision': 0.6189263738595374, 'recall': 0.7013705217600384, 'f1-score': 0.6575743913435528, 'support': 4159}
7                    : {'precision': 0.9986807387862797, 'recall': 0.6274347285536677, 'f1-score': 0.770679562229575, 'support': 4826}
8                    : {'precision': 0.8896210873146623, 'recall': 0.8951924848038313, 'f1-score': 0.892398090341535, 'support': 5429}
9                    : {'precision': 0.9054934045750542, 'recall': 0.8990384615384616, 'f1-score': 0.9022543881540637, 'support': 6032}
10                   : {'precision': 0.9983030303030302, 'recall': 0.7808115282518013, 'f1-score': 0.8762634322800298, 'support': 5274}
11                   : {'precision': 0.7540391156462585, 'recall': 0.7502115059221658, 'f1-score': 0.7521204410517388, 'support': 4728}
12                   : {'precision': 0.7155234657039711, 'recall': 0.7143200384430562, 'f1-score': 0.7149212456414573, 'support': 4162}
13                   : {'precision': 0.6661979752530933, 'recall': 0.6751211171273868, 'f1-score': 0.6706298655343241, 'support': 3509}
14                   : {'precision': 0.669364161849711, 'recall': 0.7942386831275721, 'f1-score': 0.7264742785445422, 'support': 2916}
15                   : {'precision': 0.801381124630056, 'recall': 0.9955065359477124, 'f1-score': 0.8879577336491165, 'support': 2448}
16                   : {'precision': 0.9951298701298701, 'recall': 0.9967479674796748, 'f1-score': 0.9959382615759544, 'support': 1845}
17                   : {'precision': 0.9965606190885641, 'recall': 0.9974182444061962, 'f1-score': 0.9969892473118279, 'support': 1162}
18                   : {'precision': 0.9933774834437086, 'recall': 1.0, 'f1-score': 0.9966777408637874, 'support': 600}
accuracy             : 0.8106833333333333
macro avg            : {'precision': 0.8425759087790015, 'recall': 0.8527503422888935, 'f1-score': 0.8426358223661462, 'support': 60000}
weighted avg         : {'precision': 0.8247080144279518, 'recall': 0.8106833333333333, 'f1-score': 0.8114221105818358, 'support': 60000}
TRAIN : epoch=12 acc_l: 99.73 acc_o: 82.12 loss: 1.04396: 100%|██████████| 1875/1875 [00:36<00:00, 50.70it/s]


TEST : epoch=12 acc_l: 0.32 acc_o: 0.27 loss: 0.00336:   0%|          | 0/1875 [00:00<?, ?it/s]0                    : {'precision': 0.9978073874177771, 'recall': 0.9988181664696945, 'f1-score': 0.9983125210934864, 'support': 5923}
1                    : {'precision': 0.9977751409077426, 'recall': 0.9977751409077426, 'f1-score': 0.9977751409077426, 'support': 6742}
2                    : {'precision': 0.9981528127623845, 'recall': 0.9976502181940249, 'f1-score': 0.997901452195081, 'support': 5958}
3                    : {'precision': 0.9985315712187959, 'recall': 0.9982058391779481, 'f1-score': 0.99836867862969, 'support': 6131}
4                    : {'precision': 0.997773591368385, 'recall': 0.9972612119137282, 'f1-score': 0.9975173358445338, 'support': 5842}
5                    : {'precision': 0.9970474257243034, 'recall': 0.9966795794133924, 'f1-score': 0.9968634686346863, 'support': 5421}
6                    : {'precision': 0.9978044249282216, 'recall': 0.9983102399459277, 'f1-score': 0.9980572683503675, 'support': 5918}
7                    : {'precision': 0.9960146660290132, 'recall': 0.9972865123703113, 'f1-score': 0.9966501834423355, 'support': 6265}
8                    : {'precision': 0.9960690480259785, 'recall': 0.9960690480259785, 'f1-score': 0.9960690480259785, 'support': 5851}
9                    : {'precision': 0.9957926624032313, 'recall': 0.9946209446965877, 'f1-score': 0.9952064586662182, 'support': 5949}
accuracy             : 0.9972833333333333
macro avg            : {'precision': 0.9972768730785833, 'recall': 0.9972676901115335, 'f1-score': 0.9972721555790119, 'support': 60000}
weighted avg         : {'precision': 0.9972833518485931, 'recall': 0.9972833333333333, 'f1-score': 0.9972832159789181, 'support': 60000}
0                    : {'precision': 0.998272884283247, 'recall': 0.998272884283247, 'f1-score': 0.998272884283247, 'support': 579}
1                    : {'precision': 0.9969742813918305, 'recall': 0.9977289931869796, 'f1-score': 0.99735149451381, 'support': 1321}
2                    : {'precision': 0.7665589660743134, 'recall': 0.9984218832193582, 'f1-score': 0.8672606808316198, 'support': 1901}
3                    : {'precision': 0.7830536225951699, 'recall': 0.7685817597428686, 'f1-score': 0.775750202757502, 'support': 2489}
4                    : {'precision': 0.8105475310715485, 'recall': 0.81991165477404, 'f1-score': 0.8152027027027027, 'support': 2943}
5                    : {'precision': 0.7380952380952381, 'recall': 0.8406779661016949, 'f1-score': 0.786053882725832, 'support': 3540}
6                    : {'precision': 0.7351148616971401, 'recall': 0.7484486873508354, 'f1-score': 0.7417218543046357, 'support': 4190}
7                    : {'precision': 0.8941747572815534, 'recall': 0.7643153526970954, 'f1-score': 0.8241610738255033, 'support': 4820}
8                    : {'precision': 0.8906371406371406, 'recall': 0.8566890881913304, 'f1-score': 0.8733333333333334, 'support': 5352}
9                    : {'precision': 0.9313517338995047, 'recall': 0.883073309847341, 'f1-score': 0.9065702230259192, 'support': 5961}
10                   : {'precision': 0.9701492537313433, 'recall': 0.8841131664853101, 'f1-score': 0.9251352120694564, 'support': 5514}
11                   : {'precision': 0.8582581117345695, 'recall': 0.744077834179357, 'f1-score': 0.7970998074090858, 'support': 4728}
12                   : {'precision': 0.7137440758293839, 'recall': 0.7224754137682897, 'f1-score': 0.7180832041959709, 'support': 4169}
13                   : {'precision': 0.6745562130177515, 'recall': 0.6781869688385269, 'f1-score': 0.6763667184630598, 'support': 3530}
14                   : {'precision': 0.6137295081967213, 'recall': 0.6152002738788086, 'f1-score': 0.6144640109420413, 'support': 2921}
15                   : {'precision': 0.6743782533256217, 'recall': 0.9518367346938775, 'f1-score': 0.7894380501015572, 'support': 2450}
16                   : {'precision': 0.9413827655310621, 'recall': 0.996816976127321, 'f1-score': 0.9683071373357383, 'support': 1885}
17                   : {'precision': 0.9957301451750641, 'recall': 0.9965811965811966, 'f1-score': 0.9961554891072192, 'support': 1170}
18                   : {'precision': 0.9944341372912802, 'recall': 0.9981378026070763, 'f1-score': 0.996282527881041, 'support': 537}
accuracy             : 0.8212333333333334
macro avg            : {'precision': 0.8411128147820781, 'recall': 0.8559762077133976, 'f1-score': 0.8456321310425934, 'support': 60000}
weighted avg         : {'precision': 0.8281760704479454, 'recall': 0.8212333333333334, 'f1-score': 0.8220179752079688, 'support': 60000}
TEST : epoch=12 acc_l: 99.79 acc_o: 82.11 loss: 1.03687: 100%|██████████| 1875/1875 [00:27<00:00, 68.69it/s]


0                    : {'precision': 0.9988187647654404, 'recall': 0.9993246665541111, 'f1-score': 0.9990716516161702, 'support': 5923}
1                    : {'precision': 0.9973357015985791, 'recall': 0.9994067042420647, 'f1-score': 0.9983701289079864, 'support': 6742}
2                    : {'precision': 0.998992443324937, 'recall': 0.9984894259818731, 'f1-score': 0.9987408713170485, 'support': 5958}
3                    : {'precision': 0.9988584474885844, 'recall': 0.9990213668243354, 'f1-score': 0.9989399005137405, 'support': 6131}
4                    : {'precision': 0.9976035604245121, 'recall': 0.9976035604245121, 'f1-score': 0.9976035604245121, 'support': 5842}
5                    : {'precision': 0.9987053819123358, 'recall': 0.9961261759822911, 'f1-score': 0.9974141115626154, 'support': 5421}
6                    : {'precision': 0.9959602760478034, 'recall': 0.9998310239945928, 'f1-score': 0.9978918964499536, 'support': 5918}
7                    : {'precision': 0.9988787441934968, 'recall': 0.9953711093375898, 'f1-score': 0.9971218420211064, 'support': 6265}
8                    : {'precision': 0.9979462604826288, 'recall': 0.9965817808921552, 'f1-score': 0.9972635539592953, 'support': 5851}
9                    : {'precision': 0.9958018471872376, 'recall': 0.9968061859135989, 'f1-score': 0.9963037634408602, 'support': 5949}
accuracy             : 0.9978833333333333
macro avg            : {'precision': 0.9978901427425555, 'recall': 0.9978562000147123, 'f1-score': 0.9978721280213287, 'support': 60000}
weighted avg         : {'precision': 0.9978849640522747, 'recall': 0.9978833333333333, 'f1-score': 0.9978831011356075, 'support': 60000}
0                    : {'precision': 0.9966499162479062, 'recall': 1.0, 'f1-score': 0.9983221476510068, 'support': 595}
1                    : {'precision': 0.9977186311787072, 'recall': 1.0, 'f1-score': 0.99885801294252, 'support': 1312}
2                    : {'precision': 0.7538152610441767, 'recall': 0.9984042553191489, 'f1-score': 0.8590389016018306, 'support': 1880}
3                    : {'precision': 0.5984555984555985, 'recall': 0.7524271844660194, 'f1-score': 0.6666666666666667, 'support': 2472}
4                    : {'precision': 0.7565982404692082, 'recall': 0.591743119266055, 'f1-score': 0.664092664092664, 'support': 3052}
5                    : {'precision': 0.7133479212253829, 'recall': 0.8347083926031295, 'f1-score': 0.7692711064499214, 'support': 3515}
6                    : {'precision': 0.828107502799552, 'recall': 0.7146653781106548, 'f1-score': 0.7672156659317858, 'support': 4139}
7                    : {'precision': 0.7724011039558417, 'recall': 0.8716777408637874, 'f1-score': 0.8190420446785679, 'support': 4816}
8                    : {'precision': 0.997109130330041, 'recall': 0.7689020992011889, 'f1-score': 0.868260960771974, 'support': 5383}
9                    : {'precision': 0.998326359832636, 'recall': 0.9981593038821954, 'f1-score': 0.9982428248682118, 'support': 5976}
10                   : {'precision': 0.9975470155355682, 'recall': 0.898379970544919, 'f1-score': 0.9453700116234018, 'support': 5432}
11                   : {'precision': 0.8674496644295302, 'recall': 0.7552170283806344, 'f1-score': 0.8074520303435967, 'support': 4792}
12                   : {'precision': 0.7156363636363636, 'recall': 0.7057136026775042, 'f1-score': 0.7106403466538277, 'support': 4183}
13                   : {'precision': 0.6642857142857143, 'recall': 0.6822799097065463, 'f1-score': 0.6731625835189309, 'support': 3544}
14                   : {'precision': 0.6064200976971389, 'recall': 0.60473208072373, 'f1-score': 0.6055749128919861, 'support': 2874}
15                   : {'precision': 0.6759760615559989, 'recall': 0.9983164983164983, 'f1-score': 0.8061172472387426, 'support': 2376}
16                   : {'precision': 0.9994603345925526, 'recall': 0.9956989247311828, 'f1-score': 0.9975760840290869, 'support': 1860}
17                   : {'precision': 0.9951219512195122, 'recall': 0.996742671009772, 'f1-score': 0.9959316517493897, 'support': 1228}
18                   : {'precision': 0.9982456140350877, 'recall': 0.9964973730297724, 'f1-score': 0.9973707274320772, 'support': 571}
accuracy             : 0.8211333333333334
macro avg            : {'precision': 0.8385617096066588, 'recall': 0.8507508175175126, 'f1-score': 0.8393792942703256, 'support': 60000}
weighted avg         : {'precision': 0.8336531010175168, 'recall': 0.8211333333333334, 'f1-score': 0.8220710537537342, 'support': 60000}
TRAIN : epoch=13 acc_l: 99.72 acc_o: 84.34 loss: 1.04210: 100%|██████████| 1875/1875 [00:37<00:00, 49.85it/s]


TEST : epoch=13 acc_l: 0.27 acc_o: 0.22 loss: 0.00281:   0%|          | 0/1875 [00:00<?, ?it/s]0                    : {'precision': 0.9983122362869198, 'recall': 0.9986493331082222, 'f1-score': 0.9984807562457799, 'support': 5923}
1                    : {'precision': 0.9973349126443589, 'recall': 0.999110056363097, 'f1-score': 0.998221695317131, 'support': 6742}
2                    : {'precision': 0.9983204568357407, 'recall': 0.9976502181940249, 'f1-score': 0.9979852249832102, 'support': 5958}
3                    : {'precision': 0.9982049608355091, 'recall': 0.9977165225901158, 'f1-score': 0.9979606819479566, 'support': 6131}
4                    : {'precision': 0.9965771008043813, 'recall': 0.9967476891475522, 'f1-score': 0.9966623876765084, 'support': 5842}
5                    : {'precision': 0.9970474257243034, 'recall': 0.9966795794133924, 'f1-score': 0.9968634686346863, 'support': 5421}
6                    : {'precision': 0.9979709164693946, 'recall': 0.9972963839134843, 'f1-score': 0.9976335361730899, 'support': 5918}
7                    : {'precision': 0.996488427773344, 'recall': 0.996488427773344, 'f1-score': 0.996488427773344, 'support': 6265}
8                    : {'precision': 0.9965811965811966, 'recall': 0.996410869936763, 'f1-score': 0.9964960259806854, 'support': 5851}
9                    : {'precision': 0.9952925353059852, 'recall': 0.9951252311312826, 'f1-score': 0.9952088761872742, 'support': 5949}
accuracy             : 0.9972166666666666
macro avg            : {'precision': 0.9972130169261135, 'recall': 0.9971874311571278, 'f1-score': 0.9972001080919666, 'support': 60000}
weighted avg         : {'precision': 0.9972167421405659, 'recall': 0.9972166666666666, 'f1-score': 0.9972165792057162, 'support': 60000}
0                    : {'precision': 1.0, 'recall': 1.0, 'f1-score': 1.0, 'support': 586}
1                    : {'precision': 0.9983974358974359, 'recall': 0.9983974358974359, 'f1-score': 0.9983974358974359, 'support': 1248}
2                    : {'precision': 0.7672772689425479, 'recall': 0.9978343259339469, 'f1-score': 0.867498234878795, 'support': 1847}
3                    : {'precision': 0.8172458172458172, 'recall': 0.7737611697806661, 'f1-score': 0.7949092426455248, 'support': 2462}
4                    : {'precision': 0.8267045454545454, 'recall': 0.8600985221674877, 'f1-score': 0.8430709802028006, 'support': 3045}
5                    : {'precision': 0.750609458800585, 'recall': 0.8493793103448276, 'f1-score': 0.7969457745567491, 'support': 3625}
6                    : {'precision': 0.8107423359513555, 'recall': 0.757396449704142, 'f1-score': 0.7831620166421929, 'support': 4225}
7                    : {'precision': 0.9168554599003171, 'recall': 0.8417221297836939, 'f1-score': 0.8776837996096292, 'support': 4808}
8                    : {'precision': 0.9191937869822485, 'recall': 0.9305503556720329, 'f1-score': 0.9248372093023255, 'support': 5342}
9                    : {'precision': 0.9797592997811816, 'recall': 0.9027217741935484, 'f1-score': 0.9396642182581322, 'support': 5952}
10                   : {'precision': 0.9678011004687181, 'recall': 0.8622004357298475, 'f1-score': 0.9119539126260201, 'support': 5508}
11                   : {'precision': 0.8452214452214453, 'recall': 0.7630471380471381, 'f1-score': 0.8020349480203495, 'support': 4752}
12                   : {'precision': 0.7309705042816366, 'recall': 0.7309705042816366, 'f1-score': 0.7309705042816366, 'support': 4204}
13                   : {'precision': 0.6792828685258964, 'recall': 0.6756297763939995, 'f1-score': 0.6774513977579112, 'support': 3533}
14                   : {'precision': 0.6620750293083235, 'recall': 0.7524983344437042, 'f1-score': 0.7043966323666978, 'support': 3002}
15                   : {'precision': 0.7443868739205527, 'recall': 0.9142978362324989, 'f1-score': 0.8206397562833208, 'support': 2357}
16                   : {'precision': 0.9018036072144289, 'recall': 0.995575221238938, 'f1-score': 0.9463722397476341, 'support': 1808}
17                   : {'precision': 0.9973238180196253, 'recall': 0.994661921708185, 'f1-score': 0.9959910913140312, 'support': 1124}
18                   : {'precision': 0.9964973730297724, 'recall': 0.9947552447552448, 'f1-score': 0.9956255468066492, 'support': 572}
accuracy             : 0.8434166666666667
macro avg            : {'precision': 0.8585341067866544, 'recall': 0.8734472571741565, 'f1-score': 0.8637686811156756, 'support': 60000}
weighted avg         : {'precision': 0.8493091427311729, 'recall': 0.8434166666666667, 'f1-score': 0.844319463458603, 'support': 60000}
TEST : epoch=13 acc_l: 99.80 acc_o: 82.02 loss: 1.03387: 100%|██████████| 1875/1875 [00:28<00:00, 65.59it/s]


0                    : {'precision': 0.9983142279163857, 'recall': 0.9998311666385278, 'f1-score': 0.9990721214677352, 'support': 5923}
1                    : {'precision': 0.99836867862969, 'recall': 0.9985167606051617, 'f1-score': 0.9984427141268075, 'support': 6742}
2                    : {'precision': 0.9991602284178703, 'recall': 0.9984894259818731, 'f1-score': 0.9988247145735393, 'support': 5958}
3                    : {'precision': 0.9985325289417903, 'recall': 0.998858261295058, 'f1-score': 0.9986953685583823, 'support': 6131}
4                    : {'precision': 0.9981167608286252, 'recall': 0.9979459089352961, 'f1-score': 0.9980313275699734, 'support': 5842}
5                    : {'precision': 0.9976014760147601, 'recall': 0.9974174506548608, 'f1-score': 0.9975094548473388, 'support': 5421}
6                    : {'precision': 0.9984792159513349, 'recall': 0.9984792159513349, 'f1-score': 0.9984792159513349, 'support': 5918}
7                    : {'precision': 0.9971278123504069, 'recall': 0.9974461292897047, 'f1-score': 0.9972869454197255, 'support': 6265}
8                    : {'precision': 0.9974358974358974, 'recall': 0.9972654247137241, 'f1-score': 0.9973506537902743, 'support': 5851}
9                    : {'precision': 0.9971390104341973, 'recall': 0.9959657085224407, 'f1-score': 0.9965520141283323, 'support': 5949}
accuracy             : 0.9980333333333333
macro avg            : {'precision': 0.998027583692096, 'recall': 0.9980215452587983, 'f1-score': 0.9980244530433444, 'support': 60000}
weighted avg         : {'precision': 0.9980332541784754, 'recall': 0.9980333333333333, 'f1-score': 0.9980331833163175, 'support': 60000}
0                    : {'precision': 1.0, 'recall': 1.0, 'f1-score': 1.0, 'support': 610}
1                    : {'precision': 1.0, 'recall': 0.9992254066615027, 'f1-score': 0.9996125532739248, 'support': 1291}
2                    : {'precision': 0.7519603796945935, 'recall': 0.9978094194961665, 'f1-score': 0.8576135561308544, 'support': 1826}
3                    : {'precision': 0.722200546234881, 'recall': 0.7545862209539339, 'f1-score': 0.7380382775119617, 'support': 2453}
4                    : {'precision': 0.9987347110923661, 'recall': 0.7698309492847855, 'f1-score': 0.8694694327152561, 'support': 3076}
5                    : {'precision': 0.7383781530122994, 'recall': 0.9980276134122288, 'f1-score': 0.848789839444045, 'support': 3549}
6                    : {'precision': 0.8307475317348378, 'recall': 0.7013574660633484, 'f1-score': 0.7605888429752066, 'support': 4199}
7                    : {'precision': 0.9990808823529411, 'recall': 0.8785613255203071, 'f1-score': 0.934953230835394, 'support': 4949}
8                    : {'precision': 0.8980730223123732, 'recall': 0.9983089064261556, 'f1-score': 0.9455419113721303, 'support': 5322}
9                    : {'precision': 0.8928049671292915, 'recall': 0.8136129139623898, 'f1-score': 0.8513713539399217, 'support': 6009}
10                   : {'precision': 0.8714285714285714, 'recall': 0.6761224873191809, 'f1-score': 0.7614513910927748, 'support': 5323}
11                   : {'precision': 0.7589508742714405, 'recall': 0.7364168854776812, 'f1-score': 0.7475140953357252, 'support': 4951}
12                   : {'precision': 0.6886027658559848, 'recall': 0.7122071516646116, 'f1-score': 0.7002060855861316, 'support': 4055}
13                   : {'precision': 0.6741762883694734, 'recall': 0.6830242510699002, 'f1-score': 0.6785714285714286, 'support': 3505}
14                   : {'precision': 0.6794797687861271, 'recall': 0.8037606837606838, 'f1-score': 0.7364134690681283, 'support': 2925}
15                   : {'precision': 0.7670501232539031, 'recall': 0.76079869600652, 'f1-score': 0.763911620294599, 'support': 2454}
16                   : {'precision': 0.7520033741037537, 'recall': 0.9972035794183445, 'f1-score': 0.8574176484731907, 'support': 1788}
17                   : {'precision': 0.9954504094631483, 'recall': 0.9945454545454545, 'f1-score': 0.9949977262391996, 'support': 1100}
18                   : {'precision': 0.996742671009772, 'recall': 0.9951219512195122, 'f1-score': 0.9959316517493897, 'support': 615}
accuracy             : 0.8202
macro avg            : {'precision': 0.8429402652687241, 'recall': 0.8563432295927742, 'f1-score': 0.8443365323478559, 'support': 60000}
weighted avg         : {'precision': 0.8298210866166434, 'recall': 0.8202, 'f1-score': 0.8195532847884517, 'support': 60000}
TRAIN : epoch=14 acc_l: 99.74 acc_o: 84.55 loss: 1.03895: 100%|██████████| 1875/1875 [00:37<00:00, 49.38it/s]


TEST : epoch=14 acc_l: 0.32 acc_o: 0.27 loss: 0.00322:   0%|          | 0/1875 [00:00<?, ?it/s]0                    : {'precision': 0.9978073874177771, 'recall': 0.9988181664696945, 'f1-score': 0.9983125210934864, 'support': 5923}
1                    : {'precision': 0.998220112726194, 'recall': 0.998220112726194, 'f1-score': 0.998220112726194, 'support': 6742}
2                    : {'precision': 0.9986568166554735, 'recall': 0.9983215844243034, 'f1-score': 0.9984891724022158, 'support': 5958}
3                    : {'precision': 0.9980430528375733, 'recall': 0.9982058391779481, 'f1-score': 0.9981244393704639, 'support': 6131}
4                    : {'precision': 0.9977728285077951, 'recall': 0.9969188634029442, 'f1-score': 0.9973456631560922, 'support': 5842}
5                    : {'precision': 0.9972324723247232, 'recall': 0.9970485150341265, 'f1-score': 0.9971404851950927, 'support': 5421}
6                    : {'precision': 0.9978044249282216, 'recall': 0.9983102399459277, 'f1-score': 0.9980572683503675, 'support': 5918}
7                    : {'precision': 0.9964901084875558, 'recall': 0.9969672785315243, 'f1-score': 0.9967286363999042, 'support': 6265}
8                    : {'precision': 0.9964084145715751, 'recall': 0.9957272261151939, 'f1-score': 0.9960677038810052, 'support': 5851}
9                    : {'precision': 0.9957969065232011, 'recall': 0.9956295175659775, 'f1-score': 0.9957132050096663, 'support': 5949}
accuracy             : 0.9974333333333333
macro avg            : {'precision': 0.9974232524980092, 'recall': 0.9974167343393834, 'f1-score': 0.9974199207584489, 'support': 60000}
weighted avg         : {'precision': 0.9974332844972565, 'recall': 0.9974333333333333, 'f1-score': 0.9974332372823844, 'support': 60000}
0                    : {'precision': 0.9981949458483754, 'recall': 0.9981949458483754, 'f1-score': 0.9981949458483754, 'support': 554}
1                    : {'precision': 0.999227202472952, 'recall': 0.999227202472952, 'f1-score': 0.999227202472952, 'support': 1294}
2                    : {'precision': 0.8563318777292577, 'recall': 0.9994903160040775, 'f1-score': 0.9223894637817498, 'support': 1962}
3                    : {'precision': 0.8672164948453608, 'recall': 0.865076100370218, 'f1-score': 0.8661449752883031, 'support': 2431}
4                    : {'precision': 0.8257829127394345, 'recall': 0.8925402563259941, 'f1-score': 0.8578648136449779, 'support': 3043}
5                    : {'precision': 0.7636872930990578, 'recall': 0.8379435596535345, 'f1-score': 0.799094058086864, 'support': 3579}
6                    : {'precision': 0.8047512991833704, 'recall': 0.7779904306220096, 'f1-score': 0.7911446296071037, 'support': 4180}
7                    : {'precision': 0.9974561180361231, 'recall': 0.8328377230246389, 'f1-score': 0.9077439518462784, 'support': 4708}
8                    : {'precision': 0.874351313099544, 'recall': 0.9976673246007536, 'f1-score': 0.9319477036540394, 'support': 5573}
9                    : {'precision': 0.9878260869565217, 'recall': 0.8547065708075573, 'f1-score': 0.9164575116529222, 'support': 5981}
10                   : {'precision': 0.9776612719398222, 'recall': 0.7926446128257254, 'f1-score': 0.8754847928148602, 'support': 5411}
11                   : {'precision': 0.7943710511200459, 'recall': 0.8720050441361917, 'f1-score': 0.8313796212804329, 'support': 4758}
12                   : {'precision': 0.827556073405185, 'recall': 0.683589990375361, 'f1-score': 0.7487152457504284, 'support': 4156}
13                   : {'precision': 0.6539774218955107, 'recall': 0.7056657223796033, 'f1-score': 0.6788390788935822, 'support': 3530}
14                   : {'precision': 0.637504356918787, 'recall': 0.631560773480663, 'f1-score': 0.6345186470078057, 'support': 2896}
15                   : {'precision': 0.6718990458602647, 'recall': 0.890656874745002, 'f1-score': 0.7659649122807016, 'support': 2451}
16                   : {'precision': 0.8697604790419161, 'recall': 0.9965694682675815, 'f1-score': 0.9288569144684252, 'support': 1749}
17                   : {'precision': 0.9982832618025751, 'recall': 0.994017094017094, 'f1-score': 0.9961456102783727, 'support': 1170}
18                   : {'precision': 0.9930555555555556, 'recall': 0.9965156794425087, 'f1-score': 0.9947826086956522, 'support': 574}
accuracy             : 0.8455
macro avg            : {'precision': 0.8630996874499822, 'recall': 0.8746789310210443, 'f1-score': 0.8655208782817803, 'support': 60000}
weighted avg         : {'precision': 0.8548518617162327, 'recall': 0.8455, 'f1-score': 0.8460735686625632, 'support': 60000}
TEST : epoch=14 acc_l: 99.83 acc_o: 85.14 loss: 1.02923: 100%|██████████| 1875/1875 [00:28<00:00, 65.44it/s]


0                    : {'precision': 0.9994934143870314, 'recall': 0.9993246665541111, 'f1-score': 0.9994090333474039, 'support': 5923}
1                    : {'precision': 0.9986656782802076, 'recall': 0.999110056363097, 'f1-score': 0.9988878178987173, 'support': 6742}
2                    : {'precision': 0.9994958830448664, 'recall': 0.9983215844243034, 'f1-score': 0.9989083886136535, 'support': 5958}
3                    : {'precision': 0.9986955812815914, 'recall': 0.9990213668243354, 'f1-score': 0.9988584474885845, 'support': 6131}
4                    : {'precision': 0.9974345818368394, 'recall': 0.9982882574460801, 'f1-score': 0.9978612370604842, 'support': 5842}
5                    : {'precision': 0.9988909426987062, 'recall': 0.9968640472237594, 'f1-score': 0.9978764657003047, 'support': 5421}
6                    : {'precision': 0.9978073874177771, 'recall': 0.9996620479891856, 'f1-score': 0.9987338566725754, 'support': 5918}
7                    : {'precision': 0.9976061283115225, 'recall': 0.9977653631284916, 'f1-score': 0.9976857393663714, 'support': 6265}
8                    : {'precision': 0.9982885504021907, 'recall': 0.9969236028029397, 'f1-score': 0.9976056097143835, 'support': 5851}
9                    : {'precision': 0.9964729593550554, 'recall': 0.9973104723482938, 'f1-score': 0.9968915399479124, 'support': 5949}
accuracy             : 0.9982833333333333
macro avg            : {'precision': 0.9982851107015787, 'recall': 0.9982591465104598, 'f1-score': 0.9982718135810391, 'support': 60000}
weighted avg         : {'precision': 0.998283894334589, 'recall': 0.9982833333333333, 'f1-score': 0.9982833112613132, 'support': 60000}
0                    : {'precision': 1.0, 'recall': 0.998371335504886, 'f1-score': 0.9991850040749797, 'support': 614}
1                    : {'precision': 1.0, 'recall': 1.0, 'f1-score': 1.0, 'support': 1293}
2                    : {'precision': 0.7437246963562752, 'recall': 0.9989124524197933, 'f1-score': 0.8526340218148062, 'support': 1839}
3                    : {'precision': 0.9983879634605051, 'recall': 0.7461847389558233, 'f1-score': 0.8540565387267295, 'support': 2490}
4                    : {'precision': 0.8384004382360997, 'recall': 0.998694942903752, 'f1-score': 0.9115544967242406, 'support': 3065}
5                    : {'precision': 0.8436271757439641, 'recall': 0.836348455329808, 'f1-score': 0.8399720475192173, 'support': 3593}
6                    : {'precision': 0.8504117108874657, 'recall': 0.8682858477347034, 'f1-score': 0.8592558354518142, 'support': 4282}
7                    : {'precision': 0.9985879695001412, 'recall': 0.756040196707291, 'f1-score': 0.8605500121684106, 'support': 4677}
8                    : {'precision': 0.8328690807799443, 'recall': 0.9992573338284441, 'f1-score': 0.9085077650236327, 'support': 5386}
9                    : {'precision': 0.9984627209838586, 'recall': 0.8970994475138122, 'f1-score': 0.9450709348854128, 'support': 5792}
10                   : {'precision': 0.9985978032250525, 'recall': 0.7623550401427297, 'f1-score': 0.8646297045730473, 'support': 5605}
11                   : {'precision': 0.7232013269749119, 'recall': 0.7437100213219616, 'f1-score': 0.7333123094712499, 'support': 4690}
12                   : {'precision': 0.7505697120364616, 'recall': 0.855692017005196, 'f1-score': 0.7996909833351726, 'support': 4234}
13                   : {'precision': 0.7955689828801611, 'recall': 0.6859623733719248, 'f1-score': 0.7367112216350638, 'support': 3455}
14                   : {'precision': 0.6276668960770819, 'recall': 0.6158001350438893, 'f1-score': 0.621676891615542, 'support': 2962}
15                   : {'precision': 0.6833194560088814, 'recall': 0.9983779399837794, 'f1-score': 0.8113362992255726, 'support': 2466}
16                   : {'precision': 0.9989047097480832, 'recall': 0.9961769524849808, 'f1-score': 0.9975389663658737, 'support': 1831}
17                   : {'precision': 0.9956822107081175, 'recall': 0.9956822107081175, 'f1-score': 0.9956822107081175, 'support': 1158}
18                   : {'precision': 0.9947368421052631, 'recall': 0.9982394366197183, 'f1-score': 0.9964850615114235, 'support': 568}
accuracy             : 0.8513833333333334
macro avg            : {'precision': 0.8775115629322245, 'recall': 0.8816416251358217, 'f1-score': 0.8730447528858057, 'support': 60000}
weighted avg         : {'precision': 0.8658185240235662, 'recall': 0.8513833333333334, 'f1-score': 0.8514766793297924, 'support': 60000}
TRAIN : epoch=15 acc_l: 99.75 acc_o: 85.26 loss: 1.03545: 100%|██████████| 1875/1875 [00:38<00:00, 48.43it/s]


TEST : epoch=15 acc_l: 0.27 acc_o: 0.23 loss: 0.00273:   0%|          | 0/1875 [00:00<?, ?it/s]0                    : {'precision': 0.9988181664696945, 'recall': 0.9988181664696945, 'f1-score': 0.9988181664696945, 'support': 5923}
1                    : {'precision': 0.9979243884358784, 'recall': 0.9983684366656779, 'f1-score': 0.9981463631645286, 'support': 6742}
2                    : {'precision': 0.9989914271306102, 'recall': 0.9974823766364552, 'f1-score': 0.9982363315696651, 'support': 5958}
3                    : {'precision': 0.997881010594947, 'recall': 0.998532050236503, 'f1-score': 0.9982064242621882, 'support': 6131}
4                    : {'precision': 0.997602329165953, 'recall': 0.9970900376583361, 'f1-score': 0.9973461176269154, 'support': 5842}
5                    : {'precision': 0.9972319616165344, 'recall': 0.9968640472237594, 'f1-score': 0.9970479704797048, 'support': 5421}
6                    : {'precision': 0.9984797297297298, 'recall': 0.9988171679621494, 'f1-score': 0.9986484203412739, 'support': 5918}
7                    : {'precision': 0.9963323233933982, 'recall': 0.9972865123703113, 'f1-score': 0.9968091895341417, 'support': 6265}
8                    : {'precision': 0.9965800273597811, 'recall': 0.9960690480259785, 'f1-score': 0.9963244721771092, 'support': 5851}
9                    : {'precision': 0.9951268694337086, 'recall': 0.9954614220877458, 'f1-score': 0.9952941176470589, 'support': 5949}
accuracy             : 0.9975
macro avg            : {'precision': 0.9974968233330236, 'recall': 0.9974789265336611, 'f1-score': 0.997487757327228, 'support': 60000}
weighted avg         : {'precision': 0.99750029049151, 'recall': 0.9975, 'f1-score': 0.9975000269141232, 'support': 60000}
0                    : {'precision': 1.0, 'recall': 0.9966442953020134, 'f1-score': 0.9983193277310924, 'support': 596}
1                    : {'precision': 0.9968228752978554, 'recall': 0.9992038216560509, 'f1-score': 0.9980119284294234, 'support': 1256}
2                    : {'precision': 0.8916116870876531, 'recall': 0.9984168865435357, 'f1-score': 0.9419965148120487, 'support': 1895}
3                    : {'precision': 0.9662589624630957, 'recall': 0.9087663625545418, 'f1-score': 0.936631234668847, 'support': 2521}
4                    : {'precision': 0.8334751773049646, 'recall': 0.9734923790589795, 'f1-score': 0.8980589943451016, 'support': 3018}
5                    : {'precision': 0.843010146561443, 'recall': 0.8361755661168577, 'f1-score': 0.8395789473684211, 'support': 3577}
6                    : {'precision': 0.8179723502304147, 'recall': 0.8637469586374696, 'f1-score': 0.8402366863905325, 'support': 4110}
7                    : {'precision': 0.9039260969976906, 'recall': 0.8093465674110836, 'f1-score': 0.8540257473270784, 'support': 4836}
8                    : {'precision': 0.853147996729354, 'recall': 0.9259850905218318, 'f1-score': 0.8880755809005023, 'support': 5634}
9                    : {'precision': 0.998457087753134, 'recall': 0.8697916666666666, 'f1-score': 0.9296938134147436, 'support': 5952}
10                   : {'precision': 0.9959494877293305, 'recall': 0.7760861492759005, 'f1-score': 0.872378169675467, 'support': 5386}
11                   : {'precision': 0.7668484612388001, 'recall': 0.8271008403361344, 'f1-score': 0.7958358601172427, 'support': 4760}
12                   : {'precision': 0.7931726907630522, 'recall': 0.7766035881051856, 'f1-score': 0.7848006953930212, 'support': 4069}
13                   : {'precision': 0.7208371806709757, 'recall': 0.6792343387470998, 'f1-score': 0.6994176496938929, 'support': 3448}
14                   : {'precision': 0.6192281185389387, 'recall': 0.6139391868807653, 'f1-score': 0.6165723108594956, 'support': 2927}
15                   : {'precision': 0.6695778748180495, 'recall': 0.964765100671141, 'f1-score': 0.790513833992095, 'support': 2384}
16                   : {'precision': 0.956275720164609, 'recall': 0.9983888292158969, 'f1-score': 0.976878612716763, 'support': 1862}
17                   : {'precision': 0.9966216216216216, 'recall': 0.9957805907172996, 'f1-score': 0.9962009286618826, 'support': 1185}
18                   : {'precision': 0.9982817869415808, 'recall': 0.9948630136986302, 'f1-score': 0.9965694682675814, 'support': 584}
accuracy             : 0.8525833333333334
macro avg            : {'precision': 0.8748144906796084, 'recall': 0.8846490122166885, 'f1-score': 0.8765155949876439, 'support': 60000}
weighted avg         : {'precision': 0.8611285848880993, 'recall': 0.8525833333333334, 'f1-score': 0.8531872176219043, 'support': 60000}
TEST : epoch=15 acc_l: 99.82 acc_o: 89.12 loss: 1.02901: 100%|██████████| 1875/1875 [00:29<00:00, 64.18it/s]


0                    : {'precision': 0.9981459632563627, 'recall': 0.9998311666385278, 'f1-score': 0.9989878542510122, 'support': 5923}
1                    : {'precision': 0.997779422649889, 'recall': 0.9997033521210323, 'f1-score': 0.9987404608431503, 'support': 6742}
2                    : {'precision': 0.9996637525218561, 'recall': 0.9979859013091642, 'f1-score': 0.9988241222912817, 'support': 5958}
3                    : {'precision': 0.9990204081632653, 'recall': 0.9980427336486707, 'f1-score': 0.9985313315926893, 'support': 6131}
4                    : {'precision': 0.997775876817793, 'recall': 0.9982882574460801, 'f1-score': 0.9980320013690426, 'support': 5842}
5                    : {'precision': 0.9983382570162481, 'recall': 0.9974174506548608, 'f1-score': 0.9978776414136754, 'support': 5421}
6                    : {'precision': 0.9974704890387859, 'recall': 0.9994930719837783, 'f1-score': 0.9984807562457799, 'support': 5918}
7                    : {'precision': 0.9979239859469818, 'recall': 0.9974461292897047, 'f1-score': 0.9976850003991379, 'support': 6265}
8                    : {'precision': 0.9986296676944159, 'recall': 0.996410869936763, 'f1-score': 0.9975190349901617, 'support': 5851}
9                    : {'precision': 0.9971418964357768, 'recall': 0.9969742813918305, 'f1-score': 0.9970580818693788, 'support': 5949}
accuracy             : 0.9981833333333333
macro avg            : {'precision': 0.9981889719541375, 'recall': 0.9981593214420412, 'f1-score': 0.998173628526531, 'support': 60000}
weighted avg         : {'precision': 0.9981839997527943, 'recall': 0.9981833333333333, 'f1-score': 0.9981831442362078, 'support': 60000}
0                    : {'precision': 0.9984177215189873, 'recall': 1.0, 'f1-score': 0.9992082343626286, 'support': 631}
1                    : {'precision': 0.9952, 'recall': 0.9991967871485944, 'f1-score': 0.9971943887775552, 'support': 1245}
2                    : {'precision': 0.9983202687569989, 'recall': 0.9988795518207283, 'f1-score': 0.9985998319798375, 'support': 1785}
3                    : {'precision': 0.9995953055443141, 'recall': 0.9983831851253031, 'f1-score': 0.9989888776541962, 'support': 2474}
4                    : {'precision': 0.8483606557377049, 'recall': 0.9993562922433216, 'f1-score': 0.9176887838037535, 'support': 3107}
5                    : {'precision': 0.8425900083728719, 'recall': 0.8449482227819759, 'f1-score': 0.8437674678591393, 'support': 3573}
6                    : {'precision': 0.9975062344139651, 'recall': 0.8653846153846154, 'f1-score': 0.9267602007980436, 'support': 4160}
7                    : {'precision': 0.9979343110927494, 'recall': 0.9981404958677685, 'f1-score': 0.9980373928313191, 'support': 4840}
8                    : {'precision': 0.9082706766917293, 'recall': 0.9976142411451643, 'f1-score': 0.9508483470351584, 'support': 5449}
9                    : {'precision': 0.9983246463142219, 'recall': 0.906524678837052, 'f1-score': 0.9502126151665485, 'support': 5916}
10                   : {'precision': 0.9992929530992223, 'recall': 0.7742878013148283, 'f1-score': 0.8725177487395822, 'support': 5476}
11                   : {'precision': 0.7692595362752431, 'recall': 0.8684821617057209, 'f1-score': 0.815865146256817, 'support': 4737}
12                   : {'precision': 0.8265734265734266, 'recall': 0.7160164768597044, 'f1-score': 0.7673331602181251, 'support': 4127}
13                   : {'precision': 0.6731734523145566, 'recall': 0.6748672071568353, 'f1-score': 0.6740192656708084, 'support': 3577}
14                   : {'precision': 0.665900603274921, 'recall': 0.7887036406941137, 'f1-score': 0.7221183800623053, 'support': 2939}
15                   : {'precision': 0.7934926958831341, 'recall': 0.9991638795986622, 'f1-score': 0.8845299777942264, 'support': 2392}
16                   : {'precision': 0.9994499449944995, 'recall': 0.9956164383561644, 'f1-score': 0.9975295086467197, 'support': 1825}
17                   : {'precision': 0.9982238010657194, 'recall': 0.9991111111111111, 'f1-score': 0.9986672589960018, 'support': 1125}
18                   : {'precision': 0.9967845659163987, 'recall': 0.9967845659163987, 'f1-score': 0.9967845659163987, 'support': 622}
accuracy             : 0.8912333333333333
macro avg            : {'precision': 0.9108774109389823, 'recall': 0.9169190185825297, 'f1-score': 0.9110879553983772, 'support': 60000}
weighted avg         : {'precision': 0.8993795860542512, 'recall': 0.8912333333333333, 'f1-score': 0.8917167599862083, 'support': 60000}
TRAIN : epoch=16 acc_l: 99.75 acc_o: 85.59 loss: 1.03166: 100%|██████████| 1875/1875 [00:38<00:00, 48.20it/s]


TEST : epoch=16 acc_l: 0.26 acc_o: 0.21 loss: 0.00284:   0%|          | 0/1875 [00:00<?, ?it/s]0                    : {'precision': 0.998650016874789, 'recall': 0.9991558331926389, 'f1-score': 0.9989028610009283, 'support': 5923}
1                    : {'precision': 0.9980717887867102, 'recall': 0.9980717887867102, 'f1-score': 0.9980717887867102, 'support': 6742}
2                    : {'precision': 0.9989915966386554, 'recall': 0.9976502181940249, 'f1-score': 0.9983204568357407, 'support': 5958}
3                    : {'precision': 0.9983678798759589, 'recall': 0.9977165225901158, 'f1-score': 0.9980420949583946, 'support': 6131}
4                    : {'precision': 0.9967499144714335, 'recall': 0.9974323861691201, 'f1-score': 0.9970910335386721, 'support': 5842}
5                    : {'precision': 0.9972309396344841, 'recall': 0.9964951116030253, 'f1-score': 0.9968628898320723, 'support': 5421}
6                    : {'precision': 0.9976375295308809, 'recall': 0.9989861439675566, 'f1-score': 0.9983113812901048, 'support': 5918}
7                    : {'precision': 0.9961722488038277, 'recall': 0.9969672785315243, 'f1-score': 0.9965696051057039, 'support': 6265}
8                    : {'precision': 0.9964114832535885, 'recall': 0.9965817808921552, 'f1-score': 0.9964966247970606, 'support': 5851}
9                    : {'precision': 0.9962987886944819, 'recall': 0.9954614220877458, 'f1-score': 0.9958799293702179, 'support': 5949}
accuracy             : 0.9974666666666666
macro avg            : {'precision': 0.997458218656481, 'recall': 0.9974518486014616, 'f1-score': 0.9974548665515606, 'support': 60000}
weighted avg         : {'precision': 0.9974669083220432, 'recall': 0.9974666666666666, 'f1-score': 0.9974666222901908, 'support': 60000}
0                    : {'precision': 0.9982905982905983, 'recall': 0.9982905982905983, 'f1-score': 0.9982905982905983, 'support': 585}
1                    : {'precision': 0.997639653815893, 'recall': 1.0, 'f1-score': 0.998818432453722, 'support': 1268}
2                    : {'precision': 0.9873083024854574, 'recall': 0.9989299090422686, 'f1-score': 0.9930851063829786, 'support': 1869}
3                    : {'precision': 0.9983870967741936, 'recall': 0.9888178913738019, 'f1-score': 0.9935794542536116, 'support': 2504}
4                    : {'precision': 0.8355191256830601, 'recall': 0.9967405475880052, 'f1-score': 0.909036860879905, 'support': 3068}
5                    : {'precision': 0.7304773561811505, 'recall': 0.8316610925306578, 'f1-score': 0.7777922585690081, 'support': 3588}
6                    : {'precision': 0.7910216718266254, 'recall': 0.7357811375089993, 'f1-score': 0.762402088772846, 'support': 4167}
7                    : {'precision': 0.9217267552182163, 'recall': 0.8271604938271605, 'f1-score': 0.8718869194525466, 'support': 4698}
8                    : {'precision': 0.8942112076937303, 'recall': 0.8935501755682869, 'f1-score': 0.8938805694213348, 'support': 5411}
9                    : {'precision': 0.9525, 'recall': 0.8809248554913295, 'f1-score': 0.9153153153153153, 'support': 6055}
10                   : {'precision': 0.9679067865903516, 'recall': 0.8664226898444648, 'f1-score': 0.9143574394129574, 'support': 5465}
11                   : {'precision': 0.8379464285714285, 'recall': 0.7963512940178192, 'f1-score': 0.816619534479008, 'support': 4714}
12                   : {'precision': 0.7749766573295985, 'recall': 0.7837582625118036, 'f1-score': 0.7793427230046949, 'support': 4236}
13                   : {'precision': 0.7200368776889982, 'recall': 0.6801161103047896, 'f1-score': 0.6995073891625616, 'support': 3445}
14                   : {'precision': 0.6260632868322559, 'recall': 0.6411149825783972, 'f1-score': 0.6334997417799966, 'support': 2870}
15                   : {'precision': 0.7010727747173093, 'recall': 0.9861337683523654, 'f1-score': 0.8195221148957804, 'support': 2452}
16                   : {'precision': 0.9847826086956522, 'recall': 0.9956043956043956, 'f1-score': 0.9901639344262295, 'support': 1820}
17                   : {'precision': 0.9899916597164303, 'recall': 0.9949706621961442, 'f1-score': 0.9924749163879598, 'support': 1193}
18                   : {'precision': 0.9949324324324325, 'recall': 0.9949324324324325, 'f1-score': 0.9949324324324325, 'support': 592}
accuracy             : 0.8559333333333333
macro avg            : {'precision': 0.8791995410812307, 'recall': 0.8890137525823011, 'f1-score': 0.8818162015670257, 'support': 60000}
weighted avg         : {'precision': 0.8615368609899919, 'recall': 0.8559333333333333, 'f1-score': 0.8564426798702213, 'support': 60000}
TEST : epoch=16 acc_l: 99.83 acc_o: 85.07 loss: 1.02316: 100%|██████████| 1875/1875 [00:28<00:00, 66.30it/s]


0                    : {'precision': 0.9994934999155833, 'recall': 0.9994934999155833, 'f1-score': 0.9994934999155833, 'support': 5923}
1                    : {'precision': 0.9980735032602253, 'recall': 0.9989617324236132, 'f1-score': 0.9985174203113417, 'support': 6742}
2                    : {'precision': 0.9994960524105493, 'recall': 0.9986572675394427, 'f1-score': 0.9990764839224247, 'support': 5958}
3                    : {'precision': 0.9990213668243354, 'recall': 0.9990213668243354, 'f1-score': 0.9990213668243354, 'support': 6131}
4                    : {'precision': 0.9986263736263736, 'recall': 0.9955494693598083, 'f1-score': 0.9970855477455854, 'support': 5842}
5                    : {'precision': 0.9977876106194691, 'recall': 0.9983397897066962, 'f1-score': 0.998063623789765, 'support': 5421}
6                    : {'precision': 0.9984804997467499, 'recall': 0.999324095978371, 'f1-score': 0.9989021197533992, 'support': 5918}
7                    : {'precision': 0.9974481658692185, 'recall': 0.998244213886672, 'f1-score': 0.997846031112884, 'support': 6265}
8                    : {'precision': 0.9986308403217525, 'recall': 0.9972654247137241, 'f1-score': 0.9979476654694714, 'support': 5851}
9                    : {'precision': 0.9954705586311021, 'recall': 0.9974785678265254, 'f1-score': 0.9964735516372795, 'support': 5949}
accuracy             : 0.99825
macro avg            : {'precision': 0.9982528471225358, 'recall': 0.9982335428174771, 'f1-score': 0.9982427310482069, 'support': 60000}
weighted avg         : {'precision': 0.9982509117971139, 'recall': 0.99825, 'f1-score': 0.9982499982069631, 'support': 60000}
0                    : {'precision': 0.9982993197278912, 'recall': 1.0, 'f1-score': 0.9991489361702128, 'support': 587}
1                    : {'precision': 0.9992044550517104, 'recall': 1.0, 'f1-score': 0.9996020692399523, 'support': 1256}
2                    : {'precision': 0.9994666666666666, 'recall': 0.9989339019189766, 'f1-score': 0.9992002132764596, 'support': 1876}
3                    : {'precision': 0.9987824675324676, 'recall': 0.999187982135607, 'f1-score': 0.9989851836817537, 'support': 2463}
4                    : {'precision': 0.8358874338255782, 'recall': 0.9993337774816788, 'f1-score': 0.9103322712790168, 'support': 3002}
5                    : {'precision': 0.8416184971098266, 'recall': 0.8312874678846702, 'f1-score': 0.8364210828665805, 'support': 3503}
6                    : {'precision': 0.7408906882591093, 'recall': 0.8683274021352313, 'f1-score': 0.7995630802839977, 'support': 4215}
7                    : {'precision': 0.856806406029204, 'recall': 0.7389802965671338, 'f1-score': 0.7935434616643036, 'support': 4923}
8                    : {'precision': 0.8974981329350261, 'recall': 0.887719298245614, 'f1-score': 0.8925819329681552, 'support': 5415}
9                    : {'precision': 0.9988706945228685, 'recall': 0.9056313993174061, 'f1-score': 0.949968674483129, 'support': 5860}
10                   : {'precision': 0.9971570717839374, 'recall': 0.7767115704004429, 'f1-score': 0.8732365145228216, 'support': 5419}
11                   : {'precision': 0.7439746300211416, 'recall': 0.7431890179514256, 'f1-score': 0.7435816164817749, 'support': 4735}
12                   : {'precision': 0.7452296078842524, 'recall': 0.8609496124031008, 'f1-score': 0.798920984601551, 'support': 4128}
13                   : {'precision': 0.8074722315718613, 'recall': 0.6643589033508723, 'f1-score': 0.7289577635976906, 'support': 3611}
14                   : {'precision': 0.6006578947368421, 'recall': 0.6179357021996615, 'f1-score': 0.6091743119266054, 'support': 2955}
15                   : {'precision': 0.6768136557610241, 'recall': 0.9979026845637584, 'f1-score': 0.8065773859976266, 'support': 2384}
16                   : {'precision': 0.997289972899729, 'recall': 0.997289972899729, 'f1-score': 0.997289972899729, 'support': 1845}
17                   : {'precision': 0.9949958298582152, 'recall': 0.9949958298582152, 'f1-score': 0.9949958298582152, 'support': 1199}
18                   : {'precision': 0.9951768488745981, 'recall': 0.9919871794871795, 'f1-score': 0.9935794542536115, 'support': 624}
accuracy             : 0.8506666666666667
macro avg            : {'precision': 0.880320658160629, 'recall': 0.8881432630947738, 'f1-score': 0.88029793368701, 'support': 60000}
weighted avg         : {'precision': 0.86081059970774, 'recall': 0.8506666666666667, 'f1-score': 0.851264096713, 'support': 60000}
TRAIN : epoch=17 acc_l: 99.79 acc_o: 86.21 loss: 1.02873: 100%|██████████| 1875/1875 [00:38<00:00, 48.45it/s]


TEST : epoch=17 acc_l: 0.27 acc_o: 0.22 loss: 0.00272:   0%|          | 0/1875 [00:00<?, ?it/s]0                    : {'precision': 0.9989873417721519, 'recall': 0.9993246665541111, 'f1-score': 0.9991559756920999, 'support': 5923}
1                    : {'precision': 0.9983689205219455, 'recall': 0.9986650845446455, 'f1-score': 0.9985169805724456, 'support': 6742}
2                    : {'precision': 0.998656591099916, 'recall': 0.9981537428667338, 'f1-score': 0.9984051036682614, 'support': 5958}
3                    : {'precision': 0.9988580750407831, 'recall': 0.9986951557657805, 'f1-score': 0.9987766087594813, 'support': 6131}
4                    : {'precision': 0.9979452054794521, 'recall': 0.9976035604245121, 'f1-score': 0.9977743537065572, 'support': 5842}
5                    : {'precision': 0.9970474257243034, 'recall': 0.9966795794133924, 'f1-score': 0.9968634686346863, 'support': 5421}
6                    : {'precision': 0.9984792159513349, 'recall': 0.9984792159513349, 'f1-score': 0.9984792159513349, 'support': 5918}
7                    : {'precision': 0.9972873783309398, 'recall': 0.9976057462090981, 'f1-score': 0.997446536865624, 'support': 6265}
8                    : {'precision': 0.9970935202598735, 'recall': 0.9967526918475474, 'f1-score': 0.9969230769230769, 'support': 5851}
9                    : {'precision': 0.9959684192843944, 'recall': 0.9966380904353673, 'f1-score': 0.9963031423290203, 'support': 5949}
accuracy             : 0.9978833333333333
macro avg            : {'precision': 0.9978692093465096, 'recall': 0.9978597534012522, 'f1-score': 0.9978644463102588, 'support': 60000}
weighted avg         : {'precision': 0.9978834306532146, 'recall': 0.9978833333333333, 'f1-score': 0.9978833471846894, 'support': 60000}
0                    : {'precision': 1.0, 'recall': 1.0, 'f1-score': 1.0, 'support': 578}
1                    : {'precision': 0.9976940814757879, 'recall': 1.0, 'f1-score': 0.9988457098884186, 'support': 1298}
2                    : {'precision': 0.9252789907811741, 'recall': 0.998952331063384, 'f1-score': 0.960705289672544, 'support': 1909}
3                    : {'precision': 0.968117190866006, 'recall': 0.9358600583090378, 'f1-score': 0.9517153748411689, 'support': 2401}
4                    : {'precision': 0.8380898558914948, 'recall': 0.9753370601775732, 'f1-score': 0.9015197568389057, 'support': 3041}
5                    : {'precision': 0.8228335625859697, 'recall': 0.8389901823281908, 'f1-score': 0.8308333333333333, 'support': 3565}
6                    : {'precision': 0.8375848350105313, 'recall': 0.8477025106584557, 'f1-score': 0.842613301942319, 'support': 4222}
7                    : {'precision': 0.9058591178406846, 'recall': 0.8555440414507772, 'f1-score': 0.8799829460669366, 'support': 4825}
8                    : {'precision': 0.8901996370235935, 'recall': 0.9152827019966412, 'f1-score': 0.9025669334805411, 'support': 5359}
9                    : {'precision': 0.9957089916130291, 'recall': 0.8539645366343258, 'f1-score': 0.919405673120216, 'support': 5978}
10                   : {'precision': 0.9423236514522821, 'recall': 0.8205962059620596, 'f1-score': 0.8772573635924673, 'support': 5535}
11                   : {'precision': 0.7896968412126352, 'recall': 0.7845408593091828, 'f1-score': 0.7871104067617539, 'support': 4748}
12                   : {'precision': 0.7711788435679068, 'recall': 0.8370226222330334, 'f1-score': 0.8027528286480812, 'support': 4111}
13                   : {'precision': 0.800599700149925, 'recall': 0.7485281749369218, 'f1-score': 0.7736887858591712, 'support': 3567}
14                   : {'precision': 0.6605263157894737, 'recall': 0.6027444253859349, 'f1-score': 0.6303139013452915, 'support': 2915}
15                   : {'precision': 0.6688459351904491, 'recall': 0.9787853577371048, 'f1-score': 0.7946639648767309, 'support': 2404}
16                   : {'precision': 0.9730176211453745, 'recall': 0.9971783295711061, 'f1-score': 0.9849498327759197, 'support': 1772}
17                   : {'precision': 1.0, 'recall': 0.9974468085106383, 'f1-score': 0.9987217724755006, 'support': 1175}
18                   : {'precision': 0.9966555183946488, 'recall': 0.998324958123953, 'f1-score': 0.9974895397489539, 'support': 597}
accuracy             : 0.8621166666666666
macro avg            : {'precision': 0.8833795099995246, 'recall': 0.8940421665467536, 'f1-score': 0.8860598271193817, 'support': 60000}
weighted avg         : {'precision': 0.8682442698260469, 'recall': 0.8621166666666666, 'f1-score': 0.8624707612729409, 'support': 60000}
TEST : epoch=17 acc_l: 99.85 acc_o: 83.61 loss: 1.02057: 100%|██████████| 1875/1875 [00:29<00:00, 63.71it/s]


0                    : {'precision': 0.9984825493171472, 'recall': 0.9998311666385278, 'f1-score': 0.9991564029019739, 'support': 5923}
1                    : {'precision': 0.9977787649933363, 'recall': 0.9994067042420647, 'f1-score': 0.9985920711374583, 'support': 6742}
2                    : {'precision': 0.9989934574735783, 'recall': 0.999496475327291, 'f1-score': 0.9992449030958974, 'support': 5958}
3                    : {'precision': 0.9986960065199674, 'recall': 0.9993475778828902, 'f1-score': 0.9990216859611935, 'support': 6131}
4                    : {'precision': 0.998458904109589, 'recall': 0.9981170831906881, 'f1-score': 0.9982879643896594, 'support': 5842}
5                    : {'precision': 0.9987080103359173, 'recall': 0.998155321896329, 'f1-score': 0.9984315896300396, 'support': 5421}
6                    : {'precision': 0.9989858012170385, 'recall': 0.9986481919567421, 'f1-score': 0.9988169680581377, 'support': 5918}
7                    : {'precision': 0.9979256422530717, 'recall': 0.998244213886672, 'f1-score': 0.998084902649218, 'support': 6265}
8                    : {'precision': 0.9993145990404386, 'recall': 0.9967526918475474, 'f1-score': 0.9980320013690425, 'support': 5851}
9                    : {'precision': 0.9974760222110045, 'recall': 0.9964699949571356, 'f1-score': 0.9969727547931383, 'support': 5949}
accuracy             : 0.9984666666666666
macro avg            : {'precision': 0.9984819757471091, 'recall': 0.9984469421825889, 'f1-score': 0.9984641243985758, 'support': 60000}
weighted avg         : {'precision': 0.9984669091011344, 'recall': 0.9984666666666666, 'f1-score': 0.9984664505493244, 'support': 60000}
0                    : {'precision': 1.0, 'recall': 1.0, 'f1-score': 1.0, 'support': 566}
1                    : {'precision': 0.9992325402916347, 'recall': 1.0, 'f1-score': 0.999616122840691, 'support': 1302}
2                    : {'precision': 0.7566245413779046, 'recall': 0.9994614970382337, 'f1-score': 0.8612529002320186, 'support': 1857}
3                    : {'precision': 0.9983461962513782, 'recall': 0.7520764119601329, 'f1-score': 0.8578872572240644, 'support': 2408}
4                    : {'precision': 0.8486559139784946, 'recall': 0.9987345776652958, 'f1-score': 0.9175991861648016, 'support': 3161}
5                    : {'precision': 0.7190142546508819, 'recall': 0.8409155128567392, 'f1-score': 0.7752018754884086, 'support': 3539}
6                    : {'precision': 0.7287973167225683, 'recall': 0.723940980485483, 'f1-score': 0.7263610315186246, 'support': 4202}
7                    : {'precision': 0.8494648143930768, 'recall': 0.7679637636401071, 'f1-score': 0.8066608996539792, 'support': 4857}
8                    : {'precision': 0.8912437255995538, 'recall': 0.8797944576986603, 'f1-score': 0.8854820834872552, 'support': 5449}
9                    : {'precision': 0.9976028028766365, 'recall': 0.9016666666666666, 'f1-score': 0.947211765735796, 'support': 6000}
10                   : {'precision': 0.9975389663658737, 'recall': 0.8972514296255304, 'f1-score': 0.9447411867534233, 'support': 5421}
11                   : {'precision': 0.8653190452995616, 'recall': 0.752116850127011, 'f1-score': 0.8047565118912796, 'support': 4724}
12                   : {'precision': 0.718742500599952, 'recall': 0.7271182325807235, 'f1-score': 0.7229061066859764, 'support': 4119}
13                   : {'precision': 0.6779069767441861, 'recall': 0.6915776986951364, 'f1-score': 0.6846741045214327, 'support': 3372}
14                   : {'precision': 0.6309271054493985, 'recall': 0.592949783837712, 'f1-score': 0.6113492199554261, 'support': 3007}
15                   : {'precision': 0.6671214188267395, 'recall': 0.9979591836734694, 'f1-score': 0.7996729354047425, 'support': 2450}
16                   : {'precision': 0.9988826815642458, 'recall': 0.996100278551532, 'f1-score': 0.997489539748954, 'support': 1795}
17                   : {'precision': 0.9965606190885641, 'recall': 0.9957044673539519, 'f1-score': 0.9961323592608509, 'support': 1164}
18                   : {'precision': 0.9983471074380166, 'recall': 0.9950576606260296, 'f1-score': 0.9966996699669968, 'support': 607}
accuracy             : 0.8361166666666666
macro avg            : {'precision': 0.8600172909220349, 'recall': 0.8689678659517062, 'f1-score': 0.8597734082386695, 'support': 60000}
weighted avg         : {'precision': 0.8460997553627541, 'recall': 0.8361166666666666, 'f1-score': 0.8369498445763324, 'support': 60000}
TRAIN : epoch=18 acc_l: 99.79 acc_o: 85.50 loss: 1.02387: 100%|██████████| 1875/1875 [00:39<00:00, 47.01it/s]


TEST : epoch=18 acc_l: 0.27 acc_o: 0.24 loss: 0.00256:   0%|          | 0/1875 [00:00<?, ?it/s]0                    : {'precision': 0.9989878542510121, 'recall': 0.9998311666385278, 'f1-score': 0.9994093325457767, 'support': 5923}
1                    : {'precision': 0.997924696116217, 'recall': 0.9985167606051617, 'f1-score': 0.998220640569395, 'support': 6742}
2                    : {'precision': 0.9984891724022159, 'recall': 0.9983215844243034, 'f1-score': 0.9984053713806127, 'support': 5958}
3                    : {'precision': 0.998694729972263, 'recall': 0.9983689447072256, 'f1-score': 0.9985318107667212, 'support': 6131}
4                    : {'precision': 0.9972612119137282, 'recall': 0.9972612119137282, 'f1-score': 0.9972612119137282, 'support': 5842}
5                    : {'precision': 0.9981543004798819, 'recall': 0.9976019184652278, 'f1-score': 0.9978780330288772, 'support': 5421}
6                    : {'precision': 0.9983102399459277, 'recall': 0.9983102399459277, 'f1-score': 0.9983102399459277, 'support': 5918}
7                    : {'precision': 0.9971287286648588, 'recall': 0.9977653631284916, 'f1-score': 0.9974469443114727, 'support': 6265}
8                    : {'precision': 0.9979462604826288, 'recall': 0.9965817808921552, 'f1-score': 0.9972635539592953, 'support': 5851}
9                    : {'precision': 0.9963025210084033, 'recall': 0.9964699949571356, 'f1-score': 0.9963862509454575, 'support': 5949}
accuracy             : 0.9979166666666667
macro avg            : {'precision': 0.9979199715237137, 'recall': 0.9979028965677884, 'f1-score': 0.9979113389367263, 'support': 60000}
weighted avg         : {'precision': 0.9979167076021043, 'recall': 0.9979166666666667, 'f1-score': 0.9979165925689206, 'support': 60000}
0                    : {'precision': 1.0, 'recall': 0.998330550918197, 'f1-score': 0.9991645781119466, 'support': 599}
1                    : {'precision': 1.0, 'recall': 0.9992125984251968, 'f1-score': 0.9996061441512406, 'support': 1270}
2                    : {'precision': 0.893048128342246, 'recall': 1.0, 'f1-score': 0.943502824858757, 'support': 1837}
3                    : {'precision': 0.9991220368744512, 'recall': 0.9125902165196471, 'f1-score': 0.9538977367979883, 'support': 2494}
4                    : {'precision': 0.8352745424292846, 'recall': 0.9976813514408744, 'f1-score': 0.9092830188679246, 'support': 3019}
5                    : {'precision': 0.7975846678918351, 'recall': 0.8371452190686139, 'f1-score': 0.8168862597472439, 'support': 3629}
6                    : {'precision': 0.7599015439695681, 'recall': 0.8155619596541787, 'f1-score': 0.7867485231090002, 'support': 4164}
7                    : {'precision': 0.9289099526066351, 'recall': 0.7756717350551968, 'f1-score': 0.8454029511918275, 'support': 4801}
8                    : {'precision': 0.9068421052631579, 'recall': 0.9477447744774478, 'f1-score': 0.92684238838085, 'support': 5454}
9                    : {'precision': 0.957236260823467, 'recall': 0.9084353513332215, 'f1-score': 0.9321975563586301, 'support': 5963}
10                   : {'precision': 0.9927843803056027, 'recall': 0.8450144508670521, 'f1-score': 0.9129586260733802, 'support': 5536}
11                   : {'precision': 0.842197691921726, 'recall': 0.7234913793103448, 'f1-score': 0.7783445397635057, 'support': 4640}
12                   : {'precision': 0.7107910379515318, 'recall': 0.7450275581116702, 'f1-score': 0.7275067275067274, 'support': 4173}
13                   : {'precision': 0.6997736276174307, 'recall': 0.6956399437412095, 'f1-score': 0.6977006629990126, 'support': 3555}
14                   : {'precision': 0.677245508982036, 'recall': 0.774922918807811, 'f1-score': 0.7227991691963572, 'support': 2919}
15                   : {'precision': 0.7657979293109604, 'recall': 0.8941225510629429, 'f1-score': 0.825, 'support': 2399}
16                   : {'precision': 0.8762786166585484, 'recall': 0.9983351831298557, 'f1-score': 0.9333333333333333, 'support': 1802}
17                   : {'precision': 0.998272884283247, 'recall': 0.996551724137931, 'f1-score': 0.9974115616911131, 'support': 1160}
18                   : {'precision': 0.9914965986394558, 'recall': 0.9948805460750854, 'f1-score': 0.9931856899488927, 'support': 586}
accuracy             : 0.8549666666666667
macro avg            : {'precision': 0.8753977638879571, 'recall': 0.8873873690598145, 'f1-score': 0.8790406469519859, 'support': 60000}
weighted avg         : {'precision': 0.8614181018736381, 'recall': 0.8549666666666667, 'f1-score': 0.8555016541865652, 'support': 60000}
TEST : epoch=18 acc_l: 99.85 acc_o: 84.23 loss: 1.02018: 100%|██████████| 1875/1875 [00:29<00:00, 64.43it/s]


0                    : {'precision': 0.999662276258021, 'recall': 0.9994934999155833, 'f1-score': 0.9995778809624314, 'support': 5923}
1                    : {'precision': 0.9985178597895361, 'recall': 0.9992583803025809, 'f1-score': 0.9988879828008007, 'support': 6742}
2                    : {'precision': 0.9996639784946236, 'recall': 0.9986572675394427, 'f1-score': 0.9991603694374475, 'support': 5958}
3                    : {'precision': 0.9990213668243354, 'recall': 0.9990213668243354, 'f1-score': 0.9990213668243354, 'support': 6131}
4                    : {'precision': 0.9977773978457856, 'recall': 0.9989729544676481, 'f1-score': 0.9983748182362502, 'support': 5842}
5                    : {'precision': 0.999261038241271, 'recall': 0.9977863862755949, 'f1-score': 0.9985231678050581, 'support': 5421}
6                    : {'precision': 0.9989866576591792, 'recall': 0.9994930719837783, 'f1-score': 0.9992398006588394, 'support': 5918}
7                    : {'precision': 0.9979256422530717, 'recall': 0.998244213886672, 'f1-score': 0.998084902649218, 'support': 6265}
8                    : {'precision': 0.995741056218058, 'recall': 0.9989745342676466, 'f1-score': 0.9973551744731678, 'support': 5851}
9                    : {'precision': 0.9988191632928475, 'recall': 0.9952933266095142, 'f1-score': 0.9970531278942494, 'support': 5949}
accuracy             : 0.9985333333333334
macro avg            : {'precision': 0.998537643687673, 'recall': 0.9985195002072798, 'f1-score': 0.9985278591741797, 'support': 60000}
weighted avg         : {'precision': 0.9985346335745467, 'recall': 0.9985333333333334, 'f1-score': 0.9985332844996825, 'support': 60000}
0                    : {'precision': 1.0, 'recall': 1.0, 'f1-score': 1.0, 'support': 597}
1                    : {'precision': 0.9984555984555985, 'recall': 1.0, 'f1-score': 0.9992272024729522, 'support': 1293}
2                    : {'precision': 0.9973684210526316, 'recall': 0.9994725738396625, 'f1-score': 0.9984193888303478, 'support': 1896}
3                    : {'precision': 1.0, 'recall': 0.9995920032639739, 'f1-score': 0.9997959600081616, 'support': 2451}
4                    : {'precision': 0.8414835164835165, 'recall': 0.9983702737940026, 'f1-score': 0.9132379248658318, 'support': 3068}
5                    : {'precision': 0.7200381224684298, 'recall': 0.8408458542014469, 'f1-score': 0.7757669105378, 'support': 3594}
6                    : {'precision': 0.8341779279279279, 'recall': 0.7146647370959961, 'f1-score': 0.7698103403481423, 'support': 4146}
7                    : {'precision': 0.8747716663283945, 'recall': 0.8794123648235054, 'f1-score': 0.877085877085877, 'support': 4901}
8                    : {'precision': 0.8814156012599592, 'recall': 0.8853526893727899, 'f1-score': 0.8833797585886722, 'support': 5373}
9                    : {'precision': 0.9029622063329928, 'recall': 0.8914285714285715, 'f1-score': 0.8971583220568335, 'support': 5950}
10                   : {'precision': 0.9980646944982029, 'recall': 0.6753975678203928, 'f1-score': 0.8056237446998438, 'support': 5345}
11                   : {'precision': 0.7513283740701382, 'recall': 0.7417121275702896, 'f1-score': 0.7464892830746489, 'support': 4766}
12                   : {'precision': 0.7078891257995735, 'recall': 0.7226118500604595, 'f1-score': 0.7151747247486835, 'support': 4135}
13                   : {'precision': 0.725035833731486, 'recall': 0.8566186847304544, 'f1-score': 0.7853538620778885, 'support': 3543}
14                   : {'precision': 0.7765910102358701, 'recall': 0.5899256254225829, 'f1-score': 0.670509125840538, 'support': 2958}
15                   : {'precision': 0.6644609870416323, 'recall': 0.9983429991714996, 'f1-score': 0.797881145505711, 'support': 2414}
16                   : {'precision': 0.9977464788732394, 'recall': 0.9988719684151156, 'f1-score': 0.9983089064261556, 'support': 1773}
17                   : {'precision': 0.9974511469838573, 'recall': 0.9982993197278912, 'f1-score': 0.997875053123672, 'support': 1176}
18                   : {'precision': 1.0, 'recall': 0.9919484702093397, 'f1-score': 0.9959579628132579, 'support': 621}
accuracy             : 0.8423
macro avg            : {'precision': 0.8773284585022869, 'recall': 0.8833088253130514, 'f1-score': 0.8751081838476326, 'support': 60000}
weighted avg         : {'precision': 0.8521552206475351, 'recall': 0.8423, 'f1-score': 0.8412028977078471, 'support': 60000}
TRAIN : epoch=19 acc_l: 99.79 acc_o: 84.08 loss: 1.02298: 100%|██████████| 1875/1875 [00:39<00:00, 47.37it/s]


TEST : epoch=19 acc_l: 0.59 acc_o: 0.50 loss: 0.00588:   0%|          | 6/1875 [00:00<00:31, 59.38it/s]0                    : {'precision': 0.9986504723346828, 'recall': 0.9994934999155833, 'f1-score': 0.9990718082862206, 'support': 5923}
1                    : {'precision': 0.997628927089508, 'recall': 0.9985167606051617, 'f1-score': 0.9980726464047442, 'support': 6742}
2                    : {'precision': 0.9991600873509155, 'recall': 0.9983215844243034, 'f1-score': 0.9987406598942153, 'support': 5958}
3                    : {'precision': 0.9990207279255753, 'recall': 0.9983689447072256, 'f1-score': 0.998694729972263, 'support': 6131}
4                    : {'precision': 0.997944853570817, 'recall': 0.9974323861691201, 'f1-score': 0.9976885540621522, 'support': 5842}
5                    : {'precision': 0.9974179269642198, 'recall': 0.9976019184652278, 'f1-score': 0.997509914230379, 'support': 5421}
6                    : {'precision': 0.9983105254265924, 'recall': 0.9984792159513349, 'f1-score': 0.998394863563403, 'support': 5918}
7                    : {'precision': 0.9979239859469818, 'recall': 0.9974461292897047, 'f1-score': 0.9976850003991379, 'support': 6265}
8                    : {'precision': 0.9969236028029397, 'recall': 0.9969236028029397, 'f1-score': 0.9969236028029397, 'support': 5851}
9                    : {'precision': 0.9959670643589312, 'recall': 0.996301899478904, 'f1-score': 0.9961344537815126, 'support': 5949}
accuracy             : 0.9979
macro avg            : {'precision': 0.9978948173771162, 'recall': 0.9978885941809505, 'f1-score': 0.9978916233396967, 'support': 60000}
weighted avg         : {'precision': 0.9979002133560282, 'recall': 0.9979, 'f1-score': 0.9979000219523126, 'support': 60000}
0                    : {'precision': 1.0, 'recall': 1.0, 'f1-score': 1.0, 'support': 628}
1                    : {'precision': 0.9984362783424551, 'recall': 0.9992175273865415, 'f1-score': 0.9988267500977709, 'support': 1278}
2                    : {'precision': 0.836542977923908, 'recall': 0.9977591036414566, 'f1-score': 0.9100664282064385, 'support': 1785}
3                    : {'precision': 0.931079894644425, 'recall': 0.860446247464503, 'f1-score': 0.894370651486401, 'support': 2465}
4                    : {'precision': 0.8004410143329658, 'recall': 0.9480901077375122, 'f1-score': 0.8680316843521148, 'support': 3063}
5                    : {'precision': 0.7457027300303337, 'recall': 0.8035957504767094, 'f1-score': 0.7735675888291597, 'support': 3671}
6                    : {'precision': 0.8090328915071183, 'recall': 0.7654435671156525, 'f1-score': 0.786634844868735, 'support': 4306}
7                    : {'precision': 0.8786666666666667, 'recall': 0.8235784211622579, 'f1-score': 0.8502311579400065, 'support': 4801}
8                    : {'precision': 0.9185409185409186, 'recall': 0.8983364140480592, 'f1-score': 0.9083263246425568, 'support': 5410}
9                    : {'precision': 0.9842534623411118, 'recall': 0.8832141641130405, 'f1-score': 0.9310004486316734, 'support': 5874}
10                   : {'precision': 0.9291640076579452, 'recall': 0.7946152446789158, 'f1-score': 0.8566385565797215, 'support': 5497}
11                   : {'precision': 0.7735319516407599, 'recall': 0.7559071729957806, 'f1-score': 0.7646180110968842, 'support': 4740}
12                   : {'precision': 0.727208480565371, 'recall': 0.7560617193240264, 'f1-score': 0.7413544668587896, 'support': 4083}
13                   : {'precision': 0.733493077742279, 'recall': 0.7835608646188851, 'f1-score': 0.7577007700770078, 'support': 3516}
14                   : {'precision': 0.7026388341866877, 'recall': 0.6049508307900984, 'f1-score': 0.6501457725947521, 'support': 2949}
15                   : {'precision': 0.6679019384264538, 'recall': 0.9965971926839643, 'f1-score': 0.799795186891961, 'support': 2351}
16                   : {'precision': 0.9988888888888889, 'recall': 0.9972268441486412, 'f1-score': 0.9980571745767416, 'support': 1803}
17                   : {'precision': 0.9930795847750865, 'recall': 0.9982608695652174, 'f1-score': 0.9956634865568083, 'support': 1150}
18                   : {'precision': 0.9952305246422893, 'recall': 0.9936507936507937, 'f1-score': 0.994440031771247, 'support': 630}
accuracy             : 0.84085
macro avg            : {'precision': 0.8644123222555614, 'recall': 0.8768690966106345, 'f1-score': 0.8673404913715143, 'support': 60000}
weighted avg         : {'precision': 0.8478202743392489, 'recall': 0.84085, 'f1-score': 0.8412825894079897, 'support': 60000}
TEST : epoch=19 acc_l: 99.87 acc_o: 84.22 loss: 1.01667: 100%|██████████| 1875/1875 [00:29<00:00, 64.25it/s]


  0%|          | 0/1875 [00:00<?, ?it/s]0                    : {'precision': 0.9996619908737536, 'recall': 0.9986493331082222, 'f1-score': 0.9991554054054055, 'support': 5923}
1                    : {'precision': 0.9988137603795967, 'recall': 0.999110056363097, 'f1-score': 0.9989618864007119, 'support': 6742}
2                    : {'precision': 0.9996640349403662, 'recall': 0.9988251090970124, 'f1-score': 0.9992443959365291, 'support': 5958}
3                    : {'precision': 0.9988586336213925, 'recall': 0.9991844723536127, 'f1-score': 0.9990215264187867, 'support': 6131}
4                    : {'precision': 0.9979480164158687, 'recall': 0.9989729544676481, 'f1-score': 0.9984602224123181, 'support': 5842}
5                    : {'precision': 0.9994456762749445, 'recall': 0.9977863862755949, 'f1-score': 0.9986153420105234, 'support': 5421}
6                    : {'precision': 0.998144399460189, 'recall': 0.9998310239945928, 'f1-score': 0.9989869998311667, 'support': 5918}
7                    : {'precision': 0.998245054243778, 'recall': 0.9987230646448524, 'f1-score': 0.998484002234102, 'support': 6265}
8                    : {'precision': 0.9970999658819516, 'recall': 0.9989745342676466, 'f1-score': 0.9980363698454708, 'support': 5851}
9                    : {'precision': 0.998989048020219, 'recall': 0.9966380904353673, 'f1-score': 0.9978121844496803, 'support': 5949}
accuracy             : 0.9986833333333334
macro avg            : {'precision': 0.9986870580112059, 'recall': 0.9986695025007647, 'f1-score': 0.9986778334944695, 'support': 60000}
weighted avg         : {'precision': 0.9986841639701695, 'recall': 0.9986833333333334, 'f1-score': 0.9986833134403257, 'support': 60000}
0                    : {'precision': 1.0, 'recall': 0.9982964224872232, 'f1-score': 0.9991474850809888, 'support': 587}
1                    : {'precision': 1.0, 'recall': 0.9984544049459042, 'f1-score': 0.9992266047950503, 'support': 1294}
2                    : {'precision': 0.9989264626945786, 'recall': 0.9989264626945786, 'f1-score': 0.9989264626945786, 'support': 1863}
3                    : {'precision': 0.8055467006694293, 'recall': 0.9988142292490119, 'f1-score': 0.8918298923592729, 'support': 2530}
4                    : {'precision': 0.8032089063523248, 'recall': 0.8008488410055501, 'f1-score': 0.8020271374856957, 'support': 3063}
5                    : {'precision': 0.8368045649072753, 'recall': 0.8297029702970297, 'f1-score': 0.8332386363636363, 'support': 3535}
6                    : {'precision': 0.9991928974979822, 'recall': 0.8669467787114846, 'f1-score': 0.9283839520059992, 'support': 4284}
7                    : {'precision': 0.8730190571715145, 'recall': 0.8874388254486134, 'f1-score': 0.8801698857316211, 'support': 4904}
8                    : {'precision': 0.8087697929354446, 'recall': 0.8796366389099167, 'f1-score': 0.8427159822318919, 'support': 5284}
9                    : {'precision': 0.9987817258883249, 'recall': 0.8138649900727994, 'f1-score': 0.896891238946121, 'support': 6044}
10                   : {'precision': 0.8783670033670034, 'recall': 0.7841442795416119, 'f1-score': 0.8285856079404467, 'support': 5323}
11                   : {'precision': 0.7533776538708986, 'recall': 0.7474468085106383, 'f1-score': 0.7504005126562, 'support': 4700}
12                   : {'precision': 0.7071850393700787, 'recall': 0.7002923976608187, 'f1-score': 0.7037218413320275, 'support': 4104}
13                   : {'precision': 0.656241273387322, 'recall': 0.658447744466237, 'f1-score': 0.6573426573426573, 'support': 3569}
14                   : {'precision': 0.6610216546363131, 'recall': 0.8076662143826323, 'f1-score': 0.7270229007633587, 'support': 2948}
15                   : {'precision': 0.8105475310715485, 'recall': 0.9991718426501035, 'f1-score': 0.8950296735905044, 'support': 2415}
16                   : {'precision': 0.9972268441486412, 'recall': 0.9977802441731409, 'f1-score': 0.9975034674063801, 'support': 1802}
17                   : {'precision': 1.0, 'recall': 0.9965753424657534, 'f1-score': 0.9982847341337907, 'support': 1168}
18                   : {'precision': 1.0, 'recall': 1.0, 'f1-score': 1.0, 'support': 583}
accuracy             : 0.84225
macro avg            : {'precision': 0.8730640583141411, 'recall': 0.8823397598775289, 'f1-score': 0.875286772255801, 'support': 60000}
weighted avg         : {'precision': 0.8497562045526644, 'recall': 0.84225, 'f1-score': 0.8431506661724235, 'support': 60000}
TRAIN : epoch=20 acc_l: 99.81 acc_o: 85.14 loss: 1.02123: 100%|██████████| 1875/1875 [00:38<00:00, 48.19it/s]


TEST : epoch=20 acc_l: 0.32 acc_o: 0.27 loss: 0.00307:   0%|          | 0/1875 [00:00<?, ?it/s]0                    : {'precision': 0.9994930719837783, 'recall': 0.9986493331082222, 'f1-score': 0.9990710244067225, 'support': 5923}
1                    : {'precision': 0.99836867862969, 'recall': 0.9985167606051617, 'f1-score': 0.9984427141268075, 'support': 6742}
2                    : {'precision': 0.998992443324937, 'recall': 0.9984894259818731, 'f1-score': 0.9987408713170485, 'support': 5958}
3                    : {'precision': 0.99836867862969, 'recall': 0.9982058391779481, 'f1-score': 0.9982872522632739, 'support': 6131}
4                    : {'precision': 0.9981177275838466, 'recall': 0.9984594317014721, 'f1-score': 0.9982885504021907, 'support': 5842}
5                    : {'precision': 0.998522076482542, 'recall': 0.9970485150341265, 'f1-score': 0.9977847517075873, 'support': 5421}
6                    : {'precision': 0.9983122362869198, 'recall': 0.9994930719837783, 'f1-score': 0.9989023051591658, 'support': 5918}
7                    : {'precision': 0.9968117328232106, 'recall': 0.9980845969672786, 'f1-score': 0.9974477588132078, 'support': 6265}
8                    : {'precision': 0.9964169936870841, 'recall': 0.9981199794906853, 'f1-score': 0.9972677595628415, 'support': 5851}
9                    : {'precision': 0.9978107106769957, 'recall': 0.9959657085224407, 'f1-score': 0.9968873559350551, 'support': 5949}
accuracy             : 0.9981166666666667
macro avg            : {'precision': 0.9981214350108694, 'recall': 0.9981032662572986, 'f1-score': 0.9981120343693901, 'support': 60000}
weighted avg         : {'precision': 0.9981172575368429, 'recall': 0.9981166666666667, 'f1-score': 0.9981166525691668, 'support': 60000}
0                    : {'precision': 0.9969135802469136, 'recall': 1.0, 'f1-score': 0.9984544049459042, 'support': 646}
1                    : {'precision': 1.0, 'recall': 0.9983277591973244, 'f1-score': 0.9991631799163181, 'support': 1196}
2                    : {'precision': 1.0, 'recall': 0.9994669509594882, 'f1-score': 0.9997334044254865, 'support': 1876}
3                    : {'precision': 0.7996115247652962, 'recall': 0.9983831851253031, 'f1-score': 0.8880100665108753, 'support': 2474}
4                    : {'precision': 0.7912446912773603, 'recall': 0.7964485366655706, 'f1-score': 0.793838085873484, 'support': 3041}
5                    : {'precision': 0.8499019882385886, 'recall': 0.8254011422355181, 'f1-score': 0.8374724061810156, 'support': 3677}
6                    : {'precision': 0.9997219132369299, 'recall': 0.8702493343016219, 'f1-score': 0.9305034295328071, 'support': 4131}
7                    : {'precision': 0.9064304967269927, 'recall': 0.9737331954498449, 'f1-score': 0.9388772559577226, 'support': 4835}
8                    : {'precision': 0.8777953865117099, 'recall': 0.9116678858814923, 'f1-score': 0.8944110523010674, 'support': 5468}
9                    : {'precision': 0.9990278047832004, 'recall': 0.8584795321637427, 'f1-score': 0.9234363767074047, 'support': 5985}
10                   : {'precision': 0.937318110588762, 'recall': 0.77265178077136, 'f1-score': 0.8470564434553913, 'support': 5419}
11                   : {'precision': 0.7363481228668942, 'recall': 0.7415682062298604, 'f1-score': 0.7389489457347748, 'support': 4655}
12                   : {'precision': 0.7164038597317016, 'recall': 0.7216690374585112, 'f1-score': 0.7190268099681115, 'support': 4218}
13                   : {'precision': 0.6676104190260476, 'recall': 0.6721778791334093, 'f1-score': 0.6698863636363637, 'support': 3508}
14                   : {'precision': 0.6389585947302384, 'recall': 0.6990391214824982, 'f1-score': 0.6676499508357915, 'support': 2914}
15                   : {'precision': 0.7320281431630468, 'recall': 0.9958385351643778, 'f1-score': 0.8437940761636107, 'support': 2403}
16                   : {'precision': 0.9967032967032967, 'recall': 0.9967032967032967, 'f1-score': 0.9967032967032967, 'support': 1820}
17                   : {'precision': 0.9948275862068966, 'recall': 0.9982698961937716, 'f1-score': 0.9965457685664939, 'support': 1156}
18                   : {'precision': 0.9982668977469671, 'recall': 0.9965397923875432, 'f1-score': 0.9974025974025974, 'support': 578}
accuracy             : 0.8513833333333334
macro avg            : {'precision': 0.8757427587658337, 'recall': 0.8856113193423439, 'f1-score': 0.8779428376220271, 'support': 60000}
weighted avg         : {'precision': 0.8591620529250344, 'recall': 0.8513833333333334, 'f1-score': 0.8522528345717642, 'support': 60000}
TEST : epoch=20 acc_l: 99.88 acc_o: 86.09 loss: 1.01376: 100%|██████████| 1875/1875 [00:28<00:00, 65.49it/s]


0                    : {'precision': 0.99949358541526, 'recall': 0.9996623332770556, 'f1-score': 0.9995779522241918, 'support': 5923}
1                    : {'precision': 0.9985176400830121, 'recall': 0.999110056363097, 'f1-score': 0.9988137603795966, 'support': 6742}
2                    : {'precision': 0.9996640349403662, 'recall': 0.9988251090970124, 'f1-score': 0.9992443959365291, 'support': 5958}
3                    : {'precision': 0.9991843393148451, 'recall': 0.9990213668243354, 'f1-score': 0.9991028464236197, 'support': 6131}
4                    : {'precision': 0.9984594317014721, 'recall': 0.9984594317014721, 'f1-score': 0.9984594317014721, 'support': 5842}
5                    : {'precision': 0.9994457786809533, 'recall': 0.997970854085962, 'f1-score': 0.9987077718294259, 'support': 5421}
6                    : {'precision': 0.9984812689841377, 'recall': 0.9998310239945928, 'f1-score': 0.9991556906450523, 'support': 5918}
7                    : {'precision': 0.9982444940951165, 'recall': 0.9984038308060654, 'f1-score': 0.9983241560928896, 'support': 6265}
8                    : {'precision': 0.9984620642515379, 'recall': 0.9986327123568621, 'f1-score': 0.9985473810134154, 'support': 5851}
9                    : {'precision': 0.9981506388702085, 'recall': 0.9979828542612204, 'f1-score': 0.9980667395141631, 'support': 5949}
accuracy             : 0.9988
macro avg            : {'precision': 0.9988103276336909, 'recall': 0.9987899572767676, 'f1-score': 0.9988000125760355, 'support': 60000}
weighted avg         : {'precision': 0.9988002183421062, 'recall': 0.9988, 'f1-score': 0.9987999841973687, 'support': 60000}
0                    : {'precision': 1.0, 'recall': 1.0, 'f1-score': 1.0, 'support': 594}
1                    : {'precision': 0.9992119779353822, 'recall': 1.0, 'f1-score': 0.9996058336618052, 'support': 1268}
2                    : {'precision': 0.9994612068965517, 'recall': 0.9978483055406132, 'f1-score': 0.9986541049798116, 'support': 1859}
3                    : {'precision': 0.7978654592496766, 'recall': 0.9995948136142626, 'f1-score': 0.887410071942446, 'support': 2468}
4                    : {'precision': 0.813699536730642, 'recall': 0.7973411154345007, 'f1-score': 0.8054372748116606, 'support': 3084}
5                    : {'precision': 0.8450249486351629, 'recall': 0.8361893697356956, 'f1-score': 0.8405839416058394, 'support': 3443}
6                    : {'precision': 0.9988974641675854, 'recall': 0.8730426403276319, 'f1-score': 0.9317392981102969, 'support': 4151}
7                    : {'precision': 0.9981218697829716, 'recall': 0.9983302024629513, 'f1-score': 0.9982260252530523, 'support': 4791}
8                    : {'precision': 0.8961493582263711, 'recall': 0.9998140226892319, 'f1-score': 0.9451476793248945, 'support': 5377}
9                    : {'precision': 0.9987387387387388, 'recall': 0.8975064766839378, 'f1-score': 0.9454204332253113, 'support': 6176}
10                   : {'precision': 0.9985948477751756, 'recall': 0.7781021897810219, 'f1-score': 0.8746666666666666, 'support': 5480}
11                   : {'precision': 0.7398720682302772, 'recall': 0.7302188552188552, 'f1-score': 0.7350137682694344, 'support': 4752}
12                   : {'precision': 0.702261599440429, 'recall': 0.728415961305925, 'f1-score': 0.7150997150997151, 'support': 4135}
13                   : {'precision': 0.6761987290583478, 'recall': 0.670389461626575, 'f1-score': 0.6732815645671556, 'support': 3492}
14                   : {'precision': 0.6073012623677926, 'recall': 0.596714716728126, 'f1-score': 0.6019614474129185, 'support': 2983}
15                   : {'precision': 0.6621735467565291, 'recall': 0.9991525423728813, 'f1-score': 0.7964870798851545, 'support': 2360}
16                   : {'precision': 0.9983588621444202, 'recall': 0.9983588621444202, 'f1-score': 0.9983588621444202, 'support': 1828}
17                   : {'precision': 0.9991728701406121, 'recall': 0.9991728701406121, 'f1-score': 0.9991728701406121, 'support': 1209}
18                   : {'precision': 0.9981851179673321, 'recall': 1.0, 'f1-score': 0.9990917347865577, 'support': 550}
accuracy             : 0.8609333333333333
macro avg            : {'precision': 0.8804889191707367, 'recall': 0.8894838108319599, 'f1-score': 0.8813346511519871, 'support': 60000}
weighted avg         : {'precision': 0.8705308792748228, 'recall': 0.8609333333333333, 'f1-score': 0.8618201324814195, 'support': 60000}
TRAIN : epoch=21 acc_l: 99.83 acc_o: 87.09 loss: 1.01795: 100%|██████████| 1875/1875 [00:38<00:00, 48.34it/s]


TEST : epoch=21 acc_l: 0.21 acc_o: 0.19 loss: 0.00221:   0%|          | 0/1875 [00:00<?, ?it/s]0                    : {'precision': 0.9988185654008439, 'recall': 0.9991558331926389, 'f1-score': 0.99898717083052, 'support': 5923}
1                    : {'precision': 0.9983696457684897, 'recall': 0.999110056363097, 'f1-score': 0.9987397138409074, 'support': 6742}
2                    : {'precision': 0.9991598050747773, 'recall': 0.9979859013091642, 'f1-score': 0.9985725081870854, 'support': 5958}
3                    : {'precision': 0.998694942903752, 'recall': 0.998532050236503, 'f1-score': 0.9986134899274122, 'support': 6131}
4                    : {'precision': 0.9984586401781127, 'recall': 0.9979459089352961, 'f1-score': 0.9982022087150073, 'support': 5842}
5                    : {'precision': 0.9976023607524899, 'recall': 0.9977863862755949, 'f1-score': 0.9976943650281287, 'support': 5421}
6                    : {'precision': 0.9986488768789056, 'recall': 0.9991551199729638, 'f1-score': 0.9989019342849903, 'support': 5918}
7                    : {'precision': 0.9977667889615569, 'recall': 0.9984038308060654, 'f1-score': 0.9980852082336047, 'support': 6265}
8                    : {'precision': 0.9979476654694716, 'recall': 0.9972654247137241, 'f1-score': 0.9976064284493076, 'support': 5851}
9                    : {'precision': 0.9973104723482938, 'recall': 0.9973104723482938, 'f1-score': 0.9973104723482938, 'support': 5949}
accuracy             : 0.9982833333333333
macro avg            : {'precision': 0.9982777763736694, 'recall': 0.9982650984153342, 'f1-score': 0.9982713499845257, 'support': 60000}
weighted avg         : {'precision': 0.9982834313108995, 'recall': 0.9982833333333333, 'f1-score': 0.998283293662296, 'support': 60000}
0                    : {'precision': 0.9967585089141004, 'recall': 0.9983766233766234, 'f1-score': 0.9975669099756691, 'support': 616}
1                    : {'precision': 0.9992360580595875, 'recall': 0.9992360580595875, 'f1-score': 0.9992360580595875, 'support': 1309}
2                    : {'precision': 0.9994391475042064, 'recall': 0.9988789237668162, 'f1-score': 0.9991589571068124, 'support': 1784}
3                    : {'precision': 0.8484442957510874, 'recall': 0.9992119779353822, 'f1-score': 0.9176768590555454, 'support': 2538}
4                    : {'precision': 0.8169613621480026, 'recall': 0.8466236851034951, 'f1-score': 0.8315280786535578, 'support': 2947}
5                    : {'precision': 0.8462389380530974, 'recall': 0.844603919403809, 'f1-score': 0.8454206382096975, 'support': 3623}
6                    : {'precision': 0.9981162540365985, 'recall': 0.8690253045923149, 'f1-score': 0.9291082164328657, 'support': 4268}
7                    : {'precision': 0.9499205087440381, 'recall': 0.9991638795986622, 'f1-score': 0.973920130399348, 'support': 4784}
8                    : {'precision': 0.8953191489361703, 'recall': 0.9551479934628655, 'f1-score': 0.9242663855209982, 'support': 5507}
9                    : {'precision': 0.9990685543964233, 'recall': 0.8963730569948186, 'f1-score': 0.9449387719143687, 'support': 5983}
10                   : {'precision': 0.9976937269372693, 'recall': 0.7912932138284251, 'f1-score': 0.8825869631745384, 'support': 5467}
11                   : {'precision': 0.7597210481825867, 'recall': 0.7566828036202905, 'f1-score': 0.758198882210271, 'support': 4751}
12                   : {'precision': 0.7363263445761167, 'recall': 0.793467583497053, 'f1-score': 0.7638297872340427, 'support': 4072}
13                   : {'precision': 0.7360935524652339, 'recall': 0.682790970389915, 'f1-score': 0.7084410646387833, 'support': 3411}
14                   : {'precision': 0.643211100099108, 'recall': 0.6466290269013617, 'f1-score': 0.6449155349453461, 'support': 3011}
15                   : {'precision': 0.6881563593932322, 'recall': 0.9966201943388255, 'f1-score': 0.8141501294219154, 'support': 2367}
16                   : {'precision': 0.9983407079646017, 'recall': 0.9988931931377975, 'f1-score': 0.9986168741355463, 'support': 1807}
17                   : {'precision': 0.9965957446808511, 'recall': 0.9991467576791809, 'f1-score': 0.9978696207925011, 'support': 1172}
18                   : {'precision': 0.9965694682675815, 'recall': 0.9965694682675815, 'f1-score': 0.9965694682675815, 'support': 583}
accuracy             : 0.8708666666666667
macro avg            : {'precision': 0.8895900436373628, 'recall': 0.8983544544186739, 'f1-score': 0.8909473331657355, 'support': 60000}
weighted avg         : {'precision': 0.8789290265845154, 'recall': 0.8708666666666667, 'f1-score': 0.8715626569112672, 'support': 60000}
TEST : epoch=21 acc_l: 99.89 acc_o: 87.19 loss: 1.01385: 100%|██████████| 1875/1875 [00:29<00:00, 64.05it/s]


0                    : {'precision': 0.99949358541526, 'recall': 0.9996623332770556, 'f1-score': 0.9995779522241918, 'support': 5923}
1                    : {'precision': 0.9985178597895361, 'recall': 0.9992583803025809, 'f1-score': 0.9988879828008007, 'support': 6742}
2                    : {'precision': 0.999664147774979, 'recall': 0.9991607922121517, 'f1-score': 0.9994124066146227, 'support': 5958}
3                    : {'precision': 0.9993471519503836, 'recall': 0.9986951557657805, 'f1-score': 0.9990210474791974, 'support': 6131}
4                    : {'precision': 0.9984599589322382, 'recall': 0.9988017802122561, 'f1-score': 0.9986308403217525, 'support': 5842}
5                    : {'precision': 0.9990768094534712, 'recall': 0.998155321896329, 'f1-score': 0.9986158530958753, 'support': 5421}
6                    : {'precision': 0.9986497890295358, 'recall': 0.9998310239945928, 'f1-score': 0.9992400574178839, 'support': 5918}
7                    : {'precision': 0.9982447742141376, 'recall': 0.9985634477254589, 'f1-score': 0.998404085541015, 'support': 6265}
8                    : {'precision': 0.9988034188034188, 'recall': 0.9986327123568621, 'f1-score': 0.9987180582856167, 'support': 5851}
9                    : {'precision': 0.9986543313708999, 'recall': 0.9979828542612204, 'f1-score': 0.9983184799058349, 'support': 5949}
accuracy             : 0.9988833333333333
macro avg            : {'precision': 0.9988911826733862, 'recall': 0.9988743802004288, 'f1-score': 0.9988826763686791, 'support': 60000}
weighted avg         : {'precision': 0.9988834857494803, 'recall': 0.9988833333333333, 'f1-score': 0.998883305204444, 'support': 60000}
0                    : {'precision': 0.9983079526226735, 'recall': 1.0, 'f1-score': 0.9991532599491956, 'support': 590}
1                    : {'precision': 1.0, 'recall': 0.9992175273865415, 'f1-score': 0.9996086105675147, 'support': 1278}
2                    : {'precision': 0.9989299090422686, 'recall': 1.0, 'f1-score': 0.9994646680942185, 'support': 1867}
3                    : {'precision': 0.8013457225248318, 'recall': 0.9984031936127744, 'f1-score': 0.8890863846427302, 'support': 2505}
4                    : {'precision': 0.8098360655737705, 'recall': 0.7988357050452781, 'f1-score': 0.8042982741777922, 'support': 3092}
5                    : {'precision': 0.8477401129943503, 'recall': 0.8378001116694584, 'f1-score': 0.842740803145184, 'support': 3582}
6                    : {'precision': 0.9986652429257875, 'recall': 0.8742696891797149, 'f1-score': 0.9323364485981309, 'support': 4279}
7                    : {'precision': 0.8782063111275143, 'recall': 0.999370012599748, 'f1-score': 0.9348786956094687, 'support': 4762}
8                    : {'precision': 0.8926596758817922, 'recall': 0.8762867303013289, 'f1-score': 0.8843974310540235, 'support': 5343}
9                    : {'precision': 0.9992603550295858, 'recall': 0.9057995306738184, 'f1-score': 0.9502373835062423, 'support': 5966}
10                   : {'precision': 0.9995403355550448, 'recall': 0.7955002743735138, 'f1-score': 0.885923813403952, 'support': 5467}
11                   : {'precision': 0.7576414480815088, 'recall': 0.7425111536010197, 'f1-score': 0.75, 'support': 4707}
12                   : {'precision': 0.737772031889679, 'recall': 0.8429345150172329, 'f1-score': 0.7868551074342182, 'support': 4062}
13                   : {'precision': 0.7928571428571428, 'recall': 0.6880811496196112, 'f1-score': 0.7367627093075879, 'support': 3549}
14                   : {'precision': 0.6853804502707324, 'recall': 0.7916392363396971, 'f1-score': 0.7346876431953566, 'support': 3038}
15                   : {'precision': 0.7902097902097902, 'recall': 0.9987373737373737, 'f1-score': 0.8823201338538762, 'support': 2376}
16                   : {'precision': 0.9989010989010989, 'recall': 0.9989010989010989, 'f1-score': 0.9989010989010989, 'support': 1820}
17                   : {'precision': 0.9973753280839895, 'recall': 0.9982486865148862, 'f1-score': 0.9978118161925602, 'support': 1142}
18                   : {'precision': 1.0, 'recall': 0.9982608695652174, 'f1-score': 0.999129677980853, 'support': 575}
accuracy             : 0.8718666666666667
macro avg            : {'precision': 0.8939278407142927, 'recall': 0.9023577293757007, 'f1-score': 0.8951891557691581, 'support': 60000}
weighted avg         : {'precision': 0.879991336988888, 'recall': 0.8718666666666667, 'f1-score': 0.8724417973321607, 'support': 60000}
TRAIN : epoch=22 acc_l: 99.84 acc_o: 86.93 loss: 1.01732: 100%|██████████| 1875/1875 [00:39<00:00, 47.14it/s]


TEST : epoch=22 acc_l: 0.27 acc_o: 0.24 loss: 0.00265:   0%|          | 0/1875 [00:00<?, ?it/s]0                    : {'precision': 0.9988193624557261, 'recall': 0.9998311666385278, 'f1-score': 0.9993250084373946, 'support': 5923}
1                    : {'precision': 0.9985178597895361, 'recall': 0.9992583803025809, 'f1-score': 0.9988879828008007, 'support': 6742}
2                    : {'precision': 0.9993280698807324, 'recall': 0.9984894259818731, 'f1-score': 0.9989085719083199, 'support': 5958}
3                    : {'precision': 0.998694729972263, 'recall': 0.9983689447072256, 'f1-score': 0.9985318107667212, 'support': 6131}
4                    : {'precision': 0.9981157930798219, 'recall': 0.9974323861691201, 'f1-score': 0.9977739726027397, 'support': 5842}
5                    : {'precision': 0.9979701051854586, 'recall': 0.9976019184652278, 'f1-score': 0.9977859778597786, 'support': 5421}
6                    : {'precision': 0.9984797297297298, 'recall': 0.9988171679621494, 'f1-score': 0.9986484203412739, 'support': 5918}
7                    : {'precision': 0.9979259731971921, 'recall': 0.9984038308060654, 'f1-score': 0.9981648448097025, 'support': 6265}
8                    : {'precision': 0.9982908904460776, 'recall': 0.9982908904460776, 'f1-score': 0.9982908904460776, 'support': 5851}
9                    : {'precision': 0.9974772956609486, 'recall': 0.9969742813918305, 'f1-score': 0.9972257250945776, 'support': 5949}
accuracy             : 0.9983666666666666
macro avg            : {'precision': 0.9983619809397485, 'recall': 0.9983468392870677, 'f1-score': 0.9983543205067387, 'support': 60000}
weighted avg         : {'precision': 0.9983666437037715, 'recall': 0.9983666666666666, 'f1-score': 0.9983665647513564, 'support': 60000}
0                    : {'precision': 0.9966499162479062, 'recall': 1.0, 'f1-score': 0.9983221476510068, 'support': 595}
1                    : {'precision': 1.0, 'recall': 1.0, 'f1-score': 1.0, 'support': 1289}
2                    : {'precision': 0.9972914409534128, 'recall': 1.0, 'f1-score': 0.9986438839164633, 'support': 1841}
3                    : {'precision': 0.9168870419342652, 'recall': 0.9987654320987654, 'f1-score': 0.9560764230844987, 'support': 2430}
4                    : {'precision': 0.8349767981438515, 'recall': 0.9284101902612061, 'f1-score': 0.8792182012520996, 'support': 3101}
5                    : {'precision': 0.85053581500282, 'recall': 0.8403455001393145, 'f1-score': 0.8454099509460405, 'support': 3589}
6                    : {'precision': 0.9991515837104072, 'recall': 0.8689129365469749, 'f1-score': 0.9294922388845042, 'support': 4066}
7                    : {'precision': 0.8864011745274363, 'recall': 0.9979338842975206, 'f1-score': 0.9388667508990183, 'support': 4840}
8                    : {'precision': 0.883129123468426, 'recall': 0.883129123468426, 'f1-score': 0.883129123468426, 'support': 5305}
9                    : {'precision': 0.998595258999122, 'recall': 0.9021256345177665, 'f1-score': 0.9479123260271689, 'support': 6304}
10                   : {'precision': 0.998831229546517, 'recall': 0.7830309693971046, 'f1-score': 0.8778633795582947, 'support': 5457}
11                   : {'precision': 0.7482487794523456, 'recall': 0.7525619128949615, 'f1-score': 0.7503991484832357, 'support': 4684}
12                   : {'precision': 0.7395484015202325, 'recall': 0.7923353293413173, 'f1-score': 0.7650323774283071, 'support': 4175}
13                   : {'precision': 0.7334355828220859, 'recall': 0.672763083849184, 'f1-score': 0.7017904314646316, 'support': 3554}
14                   : {'precision': 0.6268127618433774, 'recall': 0.6897163120567376, 'f1-score': 0.6567617761269627, 'support': 2820}
15                   : {'precision': 0.7287611986407168, 'recall': 0.9987298899237934, 'f1-score': 0.8426504732988034, 'support': 2362}
16                   : {'precision': 0.9994517543859649, 'recall': 0.9967195188627666, 'f1-score': 0.9980837667670407, 'support': 1829}
17                   : {'precision': 0.9991341991341991, 'recall': 0.9974070872947277, 'f1-score': 0.9982698961937715, 'support': 1157}
18                   : {'precision': 0.9983361064891847, 'recall': 0.9966777408637874, 'f1-score': 0.9975062344139651, 'support': 602}
accuracy             : 0.8693333333333333
macro avg            : {'precision': 0.8913777982538038, 'recall': 0.8999770813586503, 'f1-score': 0.8929172910454863, 'support': 60000}
weighted avg         : {'precision': 0.8774062020985526, 'recall': 0.8693333333333333, 'f1-score': 0.8701069475861767, 'support': 60000}
TEST : epoch=22 acc_l: 99.89 acc_o: 86.16 loss: 1.01468: 100%|██████████| 1875/1875 [00:29<00:00, 64.08it/s]


0                    : {'precision': 0.999493670886076, 'recall': 0.9998311666385278, 'f1-score': 0.99966239027684, 'support': 5923}
1                    : {'precision': 0.9983701289079864, 'recall': 0.9994067042420647, 'f1-score': 0.9988881476539916, 'support': 6742}
2                    : {'precision': 0.9996640349403662, 'recall': 0.9988251090970124, 'f1-score': 0.9992443959365291, 'support': 5958}
3                    : {'precision': 0.9991843393148451, 'recall': 0.9990213668243354, 'f1-score': 0.9991028464236197, 'support': 6131}
4                    : {'precision': 0.9984596953619715, 'recall': 0.998630605956864, 'f1-score': 0.9985451433461702, 'support': 5842}
5                    : {'precision': 0.999630450849963, 'recall': 0.997970854085962, 'f1-score': 0.998799963075787, 'support': 5421}
6                    : {'precision': 0.9988183659689399, 'recall': 0.9998310239945928, 'f1-score': 0.9993244384394528, 'support': 5918}
7                    : {'precision': 0.9982444940951165, 'recall': 0.9984038308060654, 'f1-score': 0.9983241560928896, 'support': 6265}
8                    : {'precision': 0.9984623270117888, 'recall': 0.9988036233122544, 'f1-score': 0.9986329460013671, 'support': 5851}
9                    : {'precision': 0.9984861227922625, 'recall': 0.9978147587829888, 'f1-score': 0.9981503278964184, 'support': 5949}
accuracy             : 0.9988666666666667
macro avg            : {'precision': 0.9988813630129316, 'recall': 0.9988539043740667, 'f1-score': 0.9988674755143065, 'support': 60000}
weighted avg         : {'precision': 0.9988669000963236, 'recall': 0.9988666666666667, 'f1-score': 0.9988666291847431, 'support': 60000}
0                    : {'precision': 1.0, 'recall': 1.0, 'f1-score': 1.0, 'support': 594}
1                    : {'precision': 0.9983525535420099, 'recall': 0.9991755976916735, 'f1-score': 0.9987639060568604, 'support': 1213}
2                    : {'precision': 0.9984084880636604, 'recall': 0.9989384288747346, 'f1-score': 0.9986733881666224, 'support': 1884}
3                    : {'precision': 0.9976133651551312, 'recall': 0.9984076433121019, 'f1-score': 0.9980103461997611, 'support': 2512}
4                    : {'precision': 0.8462377317339149, 'recall': 0.9993560849967804, 'f1-score': 0.9164452317685267, 'support': 3106}
5                    : {'precision': 0.8456546275395034, 'recall': 0.8416175231676495, 'f1-score': 0.843631245601689, 'support': 3561}
6                    : {'precision': 0.9994563740146779, 'recall': 0.8709142586451919, 'f1-score': 0.9307682571826351, 'support': 4222}
7                    : {'precision': 0.8872344289086617, 'recall': 0.9993863775823276, 'f1-score': 0.9399769141977683, 'support': 4889}
8                    : {'precision': 0.8911449520586576, 'recall': 0.8831749580771381, 'f1-score': 0.8871420550252666, 'support': 5367}
9                    : {'precision': 0.9980956008379356, 'recall': 0.899896978021978, 'f1-score': 0.9464559819413092, 'support': 5824}
10                   : {'precision': 0.9992843511450382, 'recall': 0.7693296602387512, 'f1-score': 0.8693576839265331, 'support': 5445}
11                   : {'precision': 0.7325506937033084, 'recall': 0.7406128614587829, 'f1-score': 0.7365597167078012, 'support': 4634}
12                   : {'precision': 0.7204476567964561, 'recall': 0.725522423104015, 'f1-score': 0.7229761347683669, 'support': 4259}
13                   : {'precision': 0.6687376447331262, 'recall': 0.6787044998566925, 'f1-score': 0.6736842105263157, 'support': 3489}
14                   : {'precision': 0.6184971098265896, 'recall': 0.5981585004932588, 'f1-score': 0.6081578067535942, 'support': 3041}
15                   : {'precision': 0.6642718983144515, 'recall': 0.999168744804655, 'f1-score': 0.7980082987551868, 'support': 2406}
16                   : {'precision': 0.9983443708609272, 'recall': 0.9988956377691883, 'f1-score': 0.9986199282362683, 'support': 1811}
17                   : {'precision': 0.9973262032085561, 'recall': 1.0, 'f1-score': 0.998661311914324, 'support': 1119}
18                   : {'precision': 0.9983974358974359, 'recall': 0.9983974358974359, 'f1-score': 0.9983974358974359, 'support': 624}
accuracy             : 0.8615666666666667
macro avg            : {'precision': 0.8873713413863181, 'recall': 0.8947188217890711, 'f1-score': 0.887594202822435, 'support': 60000}
weighted avg         : {'precision': 0.8708054866897958, 'recall': 0.8615666666666667, 'f1-score': 0.8623047204978581, 'support': 60000}
TRAIN : epoch=23 acc_l: 99.84 acc_o: 87.60 loss: 1.01740: 100%|██████████| 1875/1875 [00:39<00:00, 47.20it/s]


TEST : epoch=23 acc_l: 0.27 acc_o: 0.23 loss: 0.00276:   0%|          | 0/1875 [00:00<?, ?it/s]0                    : {'precision': 0.9993245525160419, 'recall': 0.9991558331926389, 'f1-score': 0.9992401857323765, 'support': 5923}
1                    : {'precision': 0.9986656782802076, 'recall': 0.999110056363097, 'f1-score': 0.9988878178987173, 'support': 6742}
2                    : {'precision': 0.9986572675394427, 'recall': 0.9986572675394427, 'f1-score': 0.9986572675394427, 'support': 5958}
3                    : {'precision': 0.9986943039007671, 'recall': 0.9980427336486707, 'f1-score': 0.9983684124653288, 'support': 6131}
4                    : {'precision': 0.9982876712328768, 'recall': 0.9979459089352961, 'f1-score': 0.9981167608286253, 'support': 5842}
5                    : {'precision': 0.9987072945521699, 'recall': 0.9976019184652278, 'f1-score': 0.998154300479882, 'support': 5421}
6                    : {'precision': 0.9991551199729638, 'recall': 0.9991551199729638, 'f1-score': 0.9991551199729638, 'support': 5918}
7                    : {'precision': 0.9977667889615569, 'recall': 0.9984038308060654, 'f1-score': 0.9980852082336047, 'support': 6265}
8                    : {'precision': 0.996756572208945, 'recall': 0.9979490685352931, 'f1-score': 0.9973524639166453, 'support': 5851}
9                    : {'precision': 0.9976462676529926, 'recall': 0.9974785678265254, 'f1-score': 0.997562410691771, 'support': 5949}
accuracy             : 0.9983666666666666
macro avg            : {'precision': 0.9983661516817964, 'recall': 0.9983500305285219, 'f1-score': 0.9983579947759358, 'support': 60000}
weighted avg         : {'precision': 0.9983669367824899, 'recall': 0.9983666666666666, 'f1-score': 0.9983667080336398, 'support': 60000}
0                    : {'precision': 1.0, 'recall': 1.0, 'f1-score': 1.0, 'support': 578}
1                    : {'precision': 1.0, 'recall': 0.9992636229749632, 'f1-score': 0.9996316758747698, 'support': 1358}
2                    : {'precision': 0.9983361064891847, 'recall': 0.9994447529150472, 'f1-score': 0.9988901220865705, 'support': 1801}
3                    : {'precision': 0.9987644151565074, 'recall': 0.9991759373712402, 'f1-score': 0.9989701338825953, 'support': 2427}
4                    : {'precision': 0.841845140032949, 'recall': 0.9986970684039088, 'f1-score': 0.9135876042908225, 'support': 3070}
5                    : {'precision': 0.8455776433304523, 'recall': 0.8354682607458013, 'f1-score': 0.8404925544100802, 'support': 3513}
6                    : {'precision': 0.9838577291381669, 'recall': 0.8690188496858385, 'f1-score': 0.9228795072500964, 'support': 4138}
7                    : {'precision': 0.891559633027523, 'recall': 0.9874009347693559, 'f1-score': 0.9370359656735127, 'support': 4921}
8                    : {'precision': 0.8925479864508844, 'recall': 0.8892013498312711, 'f1-score': 0.8908715251690458, 'support': 5334}
9                    : {'precision': 0.9985441310282075, 'recall': 0.904997525977239, 'f1-score': 0.9494722270288978, 'support': 6063}
10                   : {'precision': 0.9973427812223207, 'recall': 0.8322246858832225, 'f1-score': 0.9073327961321516, 'support': 5412}
11                   : {'precision': 0.7991111111111111, 'recall': 0.7512011698349698, 'f1-score': 0.7744158501130612, 'support': 4787}
12                   : {'precision': 0.7463445645263828, 'recall': 0.8365795724465558, 'f1-score': 0.7888901332736028, 'support': 4210}
13                   : {'precision': 0.77734375, 'recall': 0.6791808873720137, 'f1-score': 0.7249544626593806, 'support': 3516}
14                   : {'precision': 0.6037868162692848, 'recall': 0.5993734772015314, 'f1-score': 0.6015720524017467, 'support': 2873}
15                   : {'precision': 0.6758041958041958, 'recall': 0.9983471074380166, 'f1-score': 0.8060050041701419, 'support': 2420}
16                   : {'precision': 0.9972929074174337, 'recall': 0.9962141698215251, 'f1-score': 0.9967532467532466, 'support': 1849}
17                   : {'precision': 0.9982593559617058, 'recall': 0.9956597222222222, 'f1-score': 0.9969578444154715, 'support': 1152}
18                   : {'precision': 0.9982668977469671, 'recall': 0.9965397923875432, 'f1-score': 0.9974025974025974, 'support': 578}
accuracy             : 0.8759833333333333
macro avg            : {'precision': 0.8970834297217514, 'recall': 0.9035783624885404, 'f1-score': 0.897163963315147, 'support': 60000}
weighted avg         : {'precision': 0.8837650706339424, 'recall': 0.8759833333333333, 'f1-score': 0.8764929095362871, 'support': 60000}
TEST : epoch=23 acc_l: 99.88 acc_o: 88.30 loss: 1.01395: 100%|██████████| 1875/1875 [00:29<00:00, 63.19it/s]


0                    : {'precision': 0.999493670886076, 'recall': 0.9998311666385278, 'f1-score': 0.99966239027684, 'support': 5923}
1                    : {'precision': 0.9986658760747109, 'recall': 0.9992583803025809, 'f1-score': 0.9989620403321471, 'support': 6742}
2                    : {'precision': 0.9996640349403662, 'recall': 0.9988251090970124, 'f1-score': 0.9992443959365291, 'support': 5958}
3                    : {'precision': 0.9988586336213925, 'recall': 0.9991844723536127, 'f1-score': 0.9990215264187867, 'support': 6131}
4                    : {'precision': 0.9984596953619715, 'recall': 0.998630605956864, 'f1-score': 0.9985451433461702, 'support': 5842}
5                    : {'precision': 0.9988923758537936, 'recall': 0.998155321896329, 'f1-score': 0.9985237128621517, 'support': 5421}
6                    : {'precision': 0.9989868287740629, 'recall': 0.9996620479891856, 'f1-score': 0.9993243243243244, 'support': 5918}
7                    : {'precision': 0.998085513720485, 'recall': 0.9985634477254589, 'f1-score': 0.9983244235219022, 'support': 6265}
8                    : {'precision': 0.9984615384615385, 'recall': 0.9982908904460776, 'f1-score': 0.9983762071617811, 'support': 5851}
9                    : {'precision': 0.9986538785125357, 'recall': 0.9976466633047572, 'f1-score': 0.998150016818029, 'support': 5949}
accuracy             : 0.9988166666666667
macro avg            : {'precision': 0.9988222046206934, 'recall': 0.9988048105710405, 'f1-score': 0.998813418099866, 'support': 60000}
weighted avg         : {'precision': 0.9988167373803788, 'recall': 0.9988166666666667, 'f1-score': 0.998816613011944, 'support': 60000}
0                    : {'precision': 0.9983416252072969, 'recall': 1.0, 'f1-score': 0.9991701244813278, 'support': 602}
1                    : {'precision': 0.9992542878448919, 'recall': 0.9992542878448919, 'f1-score': 0.9992542878448919, 'support': 1341}
2                    : {'precision': 0.9971719457013575, 'recall': 0.9983012457531144, 'f1-score': 0.9977362761743067, 'support': 1766}
3                    : {'precision': 0.998343685300207, 'recall': 1.0, 'f1-score': 0.9991711562370493, 'support': 2411}
4                    : {'precision': 0.8435093509350935, 'recall': 0.9990228013029316, 'f1-score': 0.914703250820161, 'support': 3070}
5                    : {'precision': 0.8532731376975169, 'recall': 0.8407005838198499, 'f1-score': 0.8469402044531578, 'support': 3597}
6                    : {'precision': 0.9989074023490849, 'recall': 0.8757183908045977, 'f1-score': 0.9332652800816639, 'support': 4176}
7                    : {'precision': 0.8796855000914244, 'recall': 0.9989617940199336, 'f1-score': 0.9355371900826448, 'support': 4816}
8                    : {'precision': 0.9993729096989966, 'recall': 0.87853730246233, 'f1-score': 0.9350674750635634, 'support': 5442}
9                    : {'precision': 0.9989948065002513, 'recall': 0.9994971505196111, 'f1-score': 0.9992459153749476, 'support': 5966}
10                   : {'precision': 0.9981114258734656, 'recall': 0.7831079829598073, 'f1-score': 0.8776336274001038, 'support': 5399}
11                   : {'precision': 0.7532825074121136, 'recall': 0.7467982364056267, 'f1-score': 0.7500263574064312, 'support': 4763}
12                   : {'precision': 0.7502076411960132, 'recall': 0.8586026615969582, 'f1-score': 0.8007535460992908, 'support': 4208}
13                   : {'precision': 0.7985781990521327, 'recall': 0.6774842044801838, 'f1-score': 0.733064014916097, 'support': 3482}
14                   : {'precision': 0.6114186851211073, 'recall': 0.6051369863013699, 'f1-score': 0.6082616179001722, 'support': 2920}
15                   : {'precision': 0.6841383095499451, 'recall': 0.9995990376904571, 'f1-score': 0.812316715542522, 'support': 2494}
16                   : {'precision': 0.9978070175438597, 'recall': 0.9983543609434997, 'f1-score': 0.9980806142034548, 'support': 1823}
17                   : {'precision': 0.999140154772141, 'recall': 0.999140154772141, 'f1-score': 0.999140154772141, 'support': 1163}
18                   : {'precision': 1.0, 'recall': 1.0, 'f1-score': 1.0, 'support': 561}
accuracy             : 0.8829833333333333
macro avg            : {'precision': 0.9031336100972053, 'recall': 0.908327220088279, 'f1-score': 0.9020719899396803, 'support': 60000}
weighted avg         : {'precision': 0.8920366851020691, 'recall': 0.8829833333333333, 'f1-score': 0.8832908428062503, 'support': 60000}
TRAIN : epoch=24 acc_l: 99.83 acc_o: 88.17 loss: 1.01590: 100%|██████████| 1875/1875 [00:40<00:00, 46.59it/s]


TEST : epoch=24 acc_l: 0.32 acc_o: 0.27 loss: 0.00326:   0%|          | 0/1875 [00:00<?, ?it/s]0                    : {'precision': 0.99915611814346, 'recall': 0.9994934999155833, 'f1-score': 0.99932478055368, 'support': 5923}
1                    : {'precision': 0.9986656782802076, 'recall': 0.999110056363097, 'f1-score': 0.9988878178987173, 'support': 6742}
2                    : {'precision': 0.9984896794764222, 'recall': 0.9986572675394427, 'f1-score': 0.9985734664764622, 'support': 5958}
3                    : {'precision': 0.9988580750407831, 'recall': 0.9986951557657805, 'f1-score': 0.9987766087594813, 'support': 6131}
4                    : {'precision': 0.9979445015416238, 'recall': 0.9972612119137282, 'f1-score': 0.9976027397260274, 'support': 5842}
5                    : {'precision': 0.9981536189069424, 'recall': 0.9972329828444937, 'f1-score': 0.9976930884931254, 'support': 5421}
6                    : {'precision': 0.9984799864887688, 'recall': 0.9989861439675566, 'f1-score': 0.9987330010980656, 'support': 5918}
7                    : {'precision': 0.9980842911877394, 'recall': 0.9979249800478851, 'f1-score': 0.9980046292601165, 'support': 6265}
8                    : {'precision': 0.9979490685352931, 'recall': 0.9979490685352931, 'f1-score': 0.9979490685352931, 'support': 5851}
9                    : {'precision': 0.9973113762392876, 'recall': 0.9976466633047572, 'f1-score': 0.9974789915966388, 'support': 5949}
accuracy             : 0.9983166666666666
macro avg            : {'precision': 0.9983092393840529, 'recall': 0.9982957030197616, 'f1-score': 0.9983024192397607, 'support': 60000}
weighted avg         : {'precision': 0.9983166104467397, 'recall': 0.9983166666666666, 'f1-score': 0.9983165884514303, 'support': 60000}
0                    : {'precision': 1.0, 'recall': 1.0, 'f1-score': 1.0, 'support': 575}
1                    : {'precision': 0.9984362783424551, 'recall': 0.9992175273865415, 'f1-score': 0.9988267500977709, 'support': 1278}
2                    : {'precision': 0.9994577006507592, 'recall': 0.9994577006507592, 'f1-score': 0.9994577006507592, 'support': 1844}
3                    : {'precision': 0.9976067012365377, 'recall': 0.9996003197442046, 'f1-score': 0.9986025154721501, 'support': 2502}
4                    : {'precision': 0.831275720164609, 'recall': 0.999670075882547, 'f1-score': 0.907729179149191, 'support': 3031}
5                    : {'precision': 0.8513513513513513, 'recall': 0.8307692307692308, 'f1-score': 0.8409343715239154, 'support': 3640}
6                    : {'precision': 0.9983155530600786, 'recall': 0.8700758502569121, 'f1-score': 0.929794744411034, 'support': 4087}
7                    : {'precision': 0.8848920863309353, 'recall': 0.9985428809325562, 'f1-score': 0.9382885085574574, 'support': 4804}
8                    : {'precision': 0.9989620095495122, 'recall': 0.8852097130242825, 'f1-score': 0.9386521018238564, 'support': 5436}
9                    : {'precision': 0.998158687646468, 'recall': 0.9976576878032458, 'f1-score': 0.9979081248431094, 'support': 5977}
10                   : {'precision': 0.9986191024165708, 'recall': 0.7869060573086688, 'f1-score': 0.8802109747438889, 'support': 5514}
11                   : {'precision': 0.749678800856531, 'recall': 0.7490372272143774, 'f1-score': 0.7493578767123288, 'support': 4674}
12                   : {'precision': 0.7487091222030982, 'recall': 0.8403767205988891, 'f1-score': 0.7918989646148595, 'support': 4141}
13                   : {'precision': 0.7833935018050542, 'recall': 0.6833667334669339, 'f1-score': 0.7299694189602448, 'support': 3493}
14                   : {'precision': 0.6086803105151729, 'recall': 0.5948275862068966, 'f1-score': 0.6016742239274503, 'support': 2900}
15                   : {'precision': 0.6812533765532145, 'recall': 0.9988118811881188, 'f1-score': 0.8100208768267223, 'support': 2525}
16                   : {'precision': 0.9977924944812362, 'recall': 0.9966923925027563, 'f1-score': 0.997242140099283, 'support': 1814}
17                   : {'precision': 0.9974958263772955, 'recall': 0.9991638795986622, 'f1-score': 0.9983291562238931, 'support': 1196}
18                   : {'precision': 0.9982394366197183, 'recall': 0.9964850615114236, 'f1-score': 0.9973614775725593, 'support': 569}
accuracy             : 0.8817
macro avg            : {'precision': 0.9011746347452948, 'recall': 0.9066246592656321, 'f1-score': 0.9003294266426567, 'support': 60000}
weighted avg         : {'precision': 0.8906589549476323, 'recall': 0.8817, 'f1-score': 0.8820937586398905, 'support': 60000}
TEST : epoch=24 acc_l: 99.89 acc_o: 88.21 loss: 1.01312: 100%|██████████| 1875/1875 [00:29<00:00, 62.62it/s]


0                    : {'precision': 0.999493670886076, 'recall': 0.9998311666385278, 'f1-score': 0.99966239027684, 'support': 5923}
1                    : {'precision': 0.9985178597895361, 'recall': 0.9992583803025809, 'f1-score': 0.9988879828008007, 'support': 6742}
2                    : {'precision': 0.9996640913671482, 'recall': 0.998992950654582, 'f1-score': 0.9993284083277367, 'support': 5958}
3                    : {'precision': 0.9991844723536127, 'recall': 0.9991844723536127, 'f1-score': 0.9991844723536127, 'support': 6131}
4                    : {'precision': 0.998289136013687, 'recall': 0.9988017802122561, 'f1-score': 0.9985453923162488, 'support': 5842}
5                    : {'precision': 0.9988923758537936, 'recall': 0.998155321896329, 'f1-score': 0.9985237128621517, 'support': 5421}
6                    : {'precision': 0.9989868287740629, 'recall': 0.9996620479891856, 'f1-score': 0.9993243243243244, 'support': 5918}
7                    : {'precision': 0.998085513720485, 'recall': 0.9985634477254589, 'f1-score': 0.9983244235219022, 'support': 6265}
8                    : {'precision': 0.9986324786324786, 'recall': 0.9984618014014698, 'f1-score': 0.9985471327236988, 'support': 5851}
9                    : {'precision': 0.9988217471806093, 'recall': 0.9974785678265254, 'f1-score': 0.9981497056349873, 'support': 5949}
accuracy             : 0.99885
macro avg            : {'precision': 0.998856817457149, 'recall': 0.998838993700053, 'f1-score': 0.9988477945142302, 'support': 60000}
weighted avg         : {'precision': 0.9988501130736658, 'recall': 0.99885, 'f1-score': 0.9988499456818599, 'support': 60000}
0                    : {'precision': 0.9984, 'recall': 1.0, 'f1-score': 0.9991993594875901, 'support': 624}
1                    : {'precision': 0.9984544049459042, 'recall': 1.0, 'f1-score': 0.9992266047950503, 'support': 1292}
2                    : {'precision': 1.0, 'recall': 0.9989004947773502, 'f1-score': 0.9994499449944995, 'support': 1819}
3                    : {'precision': 0.999195494770716, 'recall': 0.9995975855130784, 'f1-score': 0.9993964996982498, 'support': 2485}
4                    : {'precision': 0.8394140409065782, 'recall': 0.9993418887792037, 'f1-score': 0.9124230133693856, 'support': 3039}
5                    : {'precision': 0.8465387823185988, 'recall': 0.84, 'f1-score': 0.8432567155912489, 'support': 3625}
6                    : {'precision': 0.9997265518184304, 'recall': 0.8688212927756654, 'f1-score': 0.9296884933248569, 'support': 4208}
7                    : {'precision': 0.8838542629924172, 'recall': 0.9993726474278545, 'f1-score': 0.9380704681519286, 'support': 4782}
8                    : {'precision': 0.9991781384836655, 'recall': 0.8859537256330844, 'f1-score': 0.9391657010428737, 'support': 5489}
9                    : {'precision': 0.9989904088844018, 'recall': 0.9986543313708999, 'f1-score': 0.9988223418573351, 'support': 5945}
10                   : {'precision': 0.9985902255639098, 'recall': 0.7841328413284133, 'f1-score': 0.8784621744522529, 'support': 5420}
11                   : {'precision': 0.7509025270758123, 'recall': 0.7453625632377741, 'f1-score': 0.7481222892203533, 'support': 4744}
12                   : {'precision': 0.7454545454545455, 'recall': 0.8560330177227482, 'f1-score': 0.796926206351, 'support': 4119}
13                   : {'precision': 0.8012088650100738, 'recall': 0.6757292551685075, 'f1-score': 0.7331387309878629, 'support': 3531}
14                   : {'precision': 0.6067726330338632, 'recall': 0.5993174061433447, 'f1-score': 0.6030219780219781, 'support': 2930}
15                   : {'precision': 0.6680767061477721, 'recall': 0.9991564740615774, 'f1-score': 0.800743620077742, 'support': 2371}
16                   : {'precision': 0.9983525535420099, 'recall': 0.9978046103183315, 'f1-score': 0.9980785067252264, 'support': 1822}
17                   : {'precision': 0.9982532751091703, 'recall': 1.0, 'f1-score': 0.9991258741258742, 'support': 1143}
18                   : {'precision': 1.0, 'recall': 0.9967320261437909, 'f1-score': 0.9983633387888707, 'support': 612}
accuracy             : 0.8821333333333333
macro avg            : {'precision': 0.9016507061083089, 'recall': 0.9076268505474538, 'f1-score': 0.9007727295296938, 'support': 60000}
weighted avg         : {'precision': 0.8916617827950408, 'recall': 0.8821333333333333, 'f1-score': 0.8825874399205897, 'support': 60000}
TRAIN : epoch=25 acc_l: 99.84 acc_o: 88.22 loss: 1.01579: 100%|██████████| 1875/1875 [00:40<00:00, 46.86it/s]


TEST : epoch=25 acc_l: 0.27 acc_o: 0.24 loss: 0.00274:   0%|          | 0/1875 [00:00<?, ?it/s]0                    : {'precision': 0.9993248945147679, 'recall': 0.9996623332770556, 'f1-score': 0.9994935854152599, 'support': 5923}
1                    : {'precision': 0.9989617324236132, 'recall': 0.9989617324236132, 'f1-score': 0.9989617324236132, 'support': 6742}
2                    : {'precision': 0.9991606513345643, 'recall': 0.998992950654582, 'f1-score': 0.9990767939571968, 'support': 5958}
3                    : {'precision': 0.998694729972263, 'recall': 0.9983689447072256, 'f1-score': 0.9985318107667212, 'support': 6131}
4                    : {'precision': 0.9982873779756808, 'recall': 0.9977747346799042, 'f1-score': 0.998030990497389, 'support': 5842}
5                    : {'precision': 0.997970479704797, 'recall': 0.9977863862755949, 'f1-score': 0.9978784244995849, 'support': 5421}
6                    : {'precision': 0.9989864864864865, 'recall': 0.999324095978371, 'f1-score': 0.9991552627132961, 'support': 5918}
7                    : {'precision': 0.9972882437390334, 'recall': 0.9979249800478851, 'f1-score': 0.9976065102920059, 'support': 6265}
8                    : {'precision': 0.9981180496150556, 'recall': 0.9970945137583319, 'f1-score': 0.9976060191518468, 'support': 5851}
9                    : {'precision': 0.9968088679879072, 'recall': 0.9976466633047572, 'f1-score': 0.9972275896832731, 'support': 5949}
accuracy             : 0.9983666666666666
macro avg            : {'precision': 0.9983601513754168, 'recall': 0.998353733510732, 'f1-score': 0.9983568719400188, 'support': 60000}
weighted avg         : {'precision': 0.9983668045157618, 'recall': 0.9983666666666666, 'f1-score': 0.9983666657186713, 'support': 60000}
0                    : {'precision': 0.998371335504886, 'recall': 1.0, 'f1-score': 0.9991850040749797, 'support': 613}
1                    : {'precision': 1.0, 'recall': 0.9984639016897081, 'f1-score': 0.9992313604919293, 'support': 1302}
2                    : {'precision': 1.0, 'recall': 0.9994728518713759, 'f1-score': 0.9997363564460848, 'support': 1897}
3                    : {'precision': 0.999194847020934, 'recall': 0.997989545637314, 'f1-score': 0.9985918326292497, 'support': 2487}
4                    : {'precision': 0.8433192686357244, 'recall': 0.9993333333333333, 'f1-score': 0.9147215865751335, 'support': 3000}
5                    : {'precision': 0.8597272474255497, 'recall': 0.8479275322536372, 'f1-score': 0.8537866224433389, 'support': 3643}
6                    : {'precision': 0.9994563740146779, 'recall': 0.8790341859909157, 'f1-score': 0.9353853981175273, 'support': 4183}
7                    : {'precision': 0.8818527229962004, 'recall': 0.9987704918032787, 'f1-score': 0.9366772364754492, 'support': 4880}
8                    : {'precision': 0.9886195995785036, 'recall': 0.8789582162263444, 'f1-score': 0.9305693314818488, 'support': 5337}
9                    : {'precision': 0.9986331795660345, 'recall': 0.9901744875487041, 'f1-score': 0.994385845525689, 'support': 5903}
10                   : {'precision': 0.9985792090930619, 'recall': 0.7709323583180987, 'f1-score': 0.8701124522851542, 'support': 5470}
11                   : {'precision': 0.7374501155219492, 'recall': 0.7478168264110756, 'f1-score': 0.7425972927241963, 'support': 4695}
12                   : {'precision': 0.7529093931837074, 'recall': 0.8669538167025604, 'f1-score': 0.8059170281392503, 'support': 4179}
13                   : {'precision': 0.8133695283338989, 'recall': 0.6767363071710898, 'f1-score': 0.7387887193712437, 'support': 3542}
14                   : {'precision': 0.615255376344086, 'recall': 0.6169137466307277, 'f1-score': 0.6160834454912516, 'support': 2968}
15                   : {'precision': 0.677519818799547, 'recall': 0.9983312473925741, 'f1-score': 0.8072187552707034, 'support': 2397}
16                   : {'precision': 0.9970845481049563, 'recall': 0.9982486865148862, 'f1-score': 0.9976662777129521, 'support': 1713}
17                   : {'precision': 0.9983471074380166, 'recall': 0.9958779884583677, 'f1-score': 0.9971110193974413, 'support': 1213}
18                   : {'precision': 0.9982698961937716, 'recall': 0.9982698961937716, 'f1-score': 0.9982698961937716, 'support': 578}
accuracy             : 0.8822
macro avg            : {'precision': 0.9030505035660792, 'recall': 0.9084318642183035, 'f1-score': 0.9018966032024839, 'support': 60000}
weighted avg         : {'precision': 0.8918553128100581, 'recall': 0.8822, 'f1-score': 0.8826280058909124, 'support': 60000}
TEST : epoch=25 acc_l: 99.89 acc_o: 87.00 loss: 1.01408: 100%|██████████| 1875/1875 [00:29<00:00, 62.85it/s]


0                    : {'precision': 0.999493670886076, 'recall': 0.9998311666385278, 'f1-score': 0.99966239027684, 'support': 5923}
1                    : {'precision': 0.9985178597895361, 'recall': 0.9992583803025809, 'f1-score': 0.9988879828008007, 'support': 6742}
2                    : {'precision': 0.9996640349403662, 'recall': 0.9988251090970124, 'f1-score': 0.9992443959365291, 'support': 5958}
3                    : {'precision': 0.9991846053489889, 'recall': 0.9993475778828902, 'f1-score': 0.9992660849710512, 'support': 6131}
4                    : {'precision': 0.9984596953619715, 'recall': 0.998630605956864, 'f1-score': 0.9985451433461702, 'support': 5842}
5                    : {'precision': 0.9994458810491319, 'recall': 0.998155321896329, 'f1-score': 0.9988001845869867, 'support': 5421}
6                    : {'precision': 0.9989866576591792, 'recall': 0.9994930719837783, 'f1-score': 0.9992398006588394, 'support': 5918}
7                    : {'precision': 0.9982447742141376, 'recall': 0.9985634477254589, 'f1-score': 0.998404085541015, 'support': 6265}
8                    : {'precision': 0.9982917663136317, 'recall': 0.9988036233122544, 'f1-score': 0.9985476292182828, 'support': 5851}
9                    : {'precision': 0.9986541049798116, 'recall': 0.9978147587829888, 'f1-score': 0.998234255444379, 'support': 5949}
accuracy             : 0.9988833333333333
macro avg            : {'precision': 0.9988943050542831, 'recall': 0.9988723063578684, 'f1-score': 0.9988831952780893, 'support': 60000}
weighted avg         : {'precision': 0.9988835027684194, 'recall': 0.9988833333333333, 'f1-score': 0.9988833104013786, 'support': 60000}
0                    : {'precision': 1.0, 'recall': 1.0, 'f1-score': 1.0, 'support': 561}
1                    : {'precision': 0.9992229992229992, 'recall': 0.9984472049689441, 'f1-score': 0.9988349514563107, 'support': 1288}
2                    : {'precision': 0.9983579638752053, 'recall': 1.0, 'f1-score': 0.9991783073130649, 'support': 1824}
3                    : {'precision': 0.999587118084228, 'recall': 1.0, 'f1-score': 0.999793516415445, 'support': 2421}
4                    : {'precision': 0.8281120044358192, 'recall': 0.9983288770053476, 'f1-score': 0.9052886801030459, 'support': 2992}
5                    : {'precision': 0.8499446290143965, 'recall': 0.8333333333333334, 'f1-score': 0.8415570175438597, 'support': 3684}
6                    : {'precision': 0.9994623655913979, 'recall': 0.873179896665101, 'f1-score': 0.9320631737277514, 'support': 4258}
7                    : {'precision': 0.890370911748584, 'recall': 0.998360991600082, 'f1-score': 0.9412787328568668, 'support': 4881}
8                    : {'precision': 0.8852118003025718, 'recall': 0.8860495930342608, 'f1-score': 0.8856304985337243, 'support': 5283}
9                    : {'precision': 0.9987261146496815, 'recall': 0.8995246680871988, 'f1-score': 0.9465332873404623, 'support': 6101}
10                   : {'precision': 0.9976308931532812, 'recall': 0.7759351391192187, 'f1-score': 0.8729270315091211, 'support': 5427}
11                   : {'precision': 0.7377652663490689, 'recall': 0.7474769635805177, 'f1-score': 0.7425893635571056, 'support': 4558}
12                   : {'precision': 0.7543934847835405, 'recall': 0.8525066602082829, 'f1-score': 0.8004548038658329, 'support': 4129}
13                   : {'precision': 0.7998685507722643, 'recall': 0.6747990019406709, 'f1-score': 0.7320300751879699, 'support': 3607}
14                   : {'precision': 0.609432082364663, 'recall': 0.5927002583979328, 'f1-score': 0.6009497298182415, 'support': 3096}
15                   : {'precision': 0.6506224066390042, 'recall': 0.99830220713073, 'f1-score': 0.7878077373974208, 'support': 2356}
16                   : {'precision': 0.9994490358126722, 'recall': 0.998898678414097, 'f1-score': 0.999173781327458, 'support': 1816}
17                   : {'precision': 0.9973427812223207, 'recall': 0.99822695035461, 'f1-score': 0.9977846699158174, 'support': 1128}
18                   : {'precision': 1.0, 'recall': 1.0, 'f1-score': 1.0, 'support': 590}
accuracy             : 0.8699666666666667
macro avg            : {'precision': 0.8945000214748262, 'recall': 0.9013721275705436, 'f1-score': 0.8938881767299737, 'support': 60000}
weighted avg         : {'precision': 0.879931475962092, 'recall': 0.8699666666666667, 'f1-score': 0.8705056051370171, 'support': 60000}
TRAIN : epoch=26 acc_l: 99.83 acc_o: 86.64 loss: 1.01513: 100%|██████████| 1875/1875 [00:40<00:00, 46.60it/s]


TEST : epoch=26 acc_l: 0.27 acc_o: 0.24 loss: 0.00268:   0%|          | 0/1875 [00:00<?, ?it/s]0                    : {'precision': 0.9988187647654404, 'recall': 0.9993246665541111, 'f1-score': 0.9990716516161702, 'support': 5923}
1                    : {'precision': 0.9982211680996146, 'recall': 0.9988134084841294, 'f1-score': 0.9985172004744959, 'support': 6742}
2                    : {'precision': 0.9988245172124265, 'recall': 0.9983215844243034, 'f1-score': 0.998572987492655, 'support': 5958}
3                    : {'precision': 0.9990212071778141, 'recall': 0.998858261295058, 'f1-score': 0.9989397275915505, 'support': 6131}
4                    : {'precision': 0.9979452054794521, 'recall': 0.9976035604245121, 'f1-score': 0.9977743537065572, 'support': 5842}
5                    : {'precision': 0.9988921713441654, 'recall': 0.997970854085962, 'f1-score': 0.9984313001753252, 'support': 5421}
6                    : {'precision': 0.9984804997467499, 'recall': 0.999324095978371, 'f1-score': 0.9989021197533992, 'support': 5918}
7                    : {'precision': 0.9979256422530717, 'recall': 0.998244213886672, 'f1-score': 0.998084902649218, 'support': 6265}
8                    : {'precision': 0.9974363356691164, 'recall': 0.9974363356691164, 'f1-score': 0.9974363356691164, 'support': 5851}
9                    : {'precision': 0.9971409350824083, 'recall': 0.9966380904353673, 'f1-score': 0.9968894493484657, 'support': 5949}
accuracy             : 0.9982666666666666
macro avg            : {'precision': 0.9982706446830258, 'recall': 0.9982535071237602, 'f1-score': 0.9982620028476953, 'support': 60000}
weighted avg         : {'precision': 0.9982666612346968, 'recall': 0.9982666666666666, 'f1-score': 0.9982665922340467, 'support': 60000}
0                    : {'precision': 1.0, 'recall': 1.0, 'f1-score': 1.0, 'support': 587}
1                    : {'precision': 0.9974853310980721, 'recall': 0.9991603694374476, 'f1-score': 0.9983221476510067, 'support': 1191}
2                    : {'precision': 0.9984025559105432, 'recall': 0.9994669509594882, 'f1-score': 0.9989344698987747, 'support': 1876}
3                    : {'precision': 0.9988085782366958, 'recall': 0.9972244250594766, 'f1-score': 0.998015873015873, 'support': 2522}
4                    : {'precision': 0.8403882448099218, 'recall': 0.9983984625240231, 'f1-score': 0.9126043039086517, 'support': 3122}
5                    : {'precision': 0.8488601182099634, 'recall': 0.8373126041088285, 'f1-score': 0.8430468204053111, 'support': 3602}
6                    : {'precision': 0.9991926803013994, 'recall': 0.8720056364490371, 'f1-score': 0.9312766491096063, 'support': 4258}
7                    : {'precision': 0.8851424450054093, 'recall': 0.9983729916615822, 'f1-score': 0.9383542005161044, 'support': 4917}
8                    : {'precision': 0.8886586284853052, 'recall': 0.8820119670905011, 'f1-score': 0.8853228228228228, 'support': 5348}
9                    : {'precision': 0.9992629445365764, 'recall': 0.9024796139124647, 'f1-score': 0.9484085344526059, 'support': 6009}
10                   : {'precision': 0.9985369422092173, 'recall': 0.7758620689655172, 'f1-score': 0.8732274229662011, 'support': 5278}
11                   : {'precision': 0.748083475298126, 'recall': 0.7466524973432519, 'f1-score': 0.74736730135092, 'support': 4705}
12                   : {'precision': 0.7254495159059474, 'recall': 0.7623546511627907, 'f1-score': 0.7434443656980864, 'support': 4128}
13                   : {'precision': 0.7108291531425199, 'recall': 0.6869118905047049, 'f1-score': 0.6986658932714618, 'support': 3507}
14                   : {'precision': 0.6177391304347826, 'recall': 0.6016260162601627, 'f1-score': 0.6095761112064526, 'support': 2952}
15                   : {'precision': 0.6672335600907029, 'recall': 0.9970351545955104, 'f1-score': 0.7994566140261505, 'support': 2361}
16                   : {'precision': 0.9972206781545303, 'recall': 0.998330550918197, 'f1-score': 0.9977753058954394, 'support': 1797}
17                   : {'precision': 0.999163179916318, 'recall': 0.9974937343358395, 'f1-score': 0.9983277591973244, 'support': 1197}
18                   : {'precision': 0.9968944099378882, 'recall': 0.9984447900466563, 'f1-score': 0.9976689976689977, 'support': 643}
accuracy             : 0.8664333333333334
macro avg            : {'precision': 0.8903869248254694, 'recall': 0.8974286513334463, 'f1-score': 0.890515557529568, 'support': 60000}
weighted avg         : {'precision': 0.8752945071883694, 'recall': 0.8664333333333334, 'f1-score': 0.867111092656464, 'support': 60000}
TEST : epoch=26 acc_l: 99.88 acc_o: 86.36 loss: 1.01166: 100%|██████████| 1875/1875 [00:30<00:00, 62.30it/s]


0                    : {'precision': 0.99949358541526, 'recall': 0.9996623332770556, 'f1-score': 0.9995779522241918, 'support': 5923}
1                    : {'precision': 0.9979271542789458, 'recall': 0.9997033521210323, 'f1-score': 0.998814463544754, 'support': 6742}
2                    : {'precision': 0.9996640913671482, 'recall': 0.998992950654582, 'f1-score': 0.9993284083277367, 'support': 5958}
3                    : {'precision': 0.9990215264187867, 'recall': 0.9991844723536127, 'f1-score': 0.9991029927423958, 'support': 6131}
4                    : {'precision': 0.9984594317014721, 'recall': 0.9984594317014721, 'f1-score': 0.9984594317014721, 'support': 5842}
5                    : {'precision': 0.9990766389658357, 'recall': 0.997970854085962, 'f1-score': 0.9985234403839055, 'support': 5421}
6                    : {'precision': 0.9986495611073599, 'recall': 0.9996620479891856, 'f1-score': 0.9991555480493162, 'support': 5918}
7                    : {'precision': 0.9985629889829155, 'recall': 0.998244213886672, 'f1-score': 0.9984035759897829, 'support': 6265}
8                    : {'precision': 0.9988030095759234, 'recall': 0.9982908904460776, 'f1-score': 0.9985468843490897, 'support': 5851}
9                    : {'precision': 0.9986541049798116, 'recall': 0.9978147587829888, 'f1-score': 0.998234255444379, 'support': 5949}
accuracy             : 0.9988166666666667
macro avg            : {'precision': 0.9988312092793459, 'recall': 0.998798530529864, 'f1-score': 0.9988146952757024, 'support': 60000}
weighted avg         : {'precision': 0.9988169063900172, 'recall': 0.9988166666666667, 'f1-score': 0.9988166057130171, 'support': 60000}
0                    : {'precision': 1.0, 'recall': 1.0, 'f1-score': 1.0, 'support': 605}
1                    : {'precision': 1.0, 'recall': 0.9992050874403816, 'f1-score': 0.9996023856858848, 'support': 1258}
2                    : {'precision': 0.9989270386266095, 'recall': 0.9994632313472893, 'f1-score': 0.9991950630533941, 'support': 1863}
3                    : {'precision': 0.9991820040899796, 'recall': 0.9995908346972177, 'f1-score': 0.9993863775823276, 'support': 2444}
4                    : {'precision': 0.843682505399568, 'recall': 0.9993604093380236, 'f1-score': 0.9149465671204802, 'support': 3127}
5                    : {'precision': 0.8486134691567628, 'recall': 0.8381777529346003, 'f1-score': 0.843363329583802, 'support': 3578}
6                    : {'precision': 0.9989035087719298, 'recall': 0.8715618273140397, 'f1-score': 0.9308979435432367, 'support': 4181}
7                    : {'precision': 0.882073749543629, 'recall': 0.998966301426504, 'f1-score': 0.9368880271449346, 'support': 4837}
8                    : {'precision': 0.8896590483327089, 'recall': 0.8814031180400891, 'f1-score': 0.8855118403878427, 'support': 5388}
9                    : {'precision': 0.9991034606419221, 'recall': 0.90439863658497, 'f1-score': 0.9493951269381496, 'support': 6161}
10                   : {'precision': 0.9990236758603857, 'recall': 0.774016641452345, 'f1-score': 0.8722429408630793, 'support': 5288}
11                   : {'precision': 0.7509410288582183, 'recall': 0.7473465140478668, 'f1-score': 0.7491394596849902, 'support': 4805}
12                   : {'precision': 0.7044071098125152, 'recall': 0.7209070520807376, 'f1-score': 0.7125615763546798, 'support': 4013}
13                   : {'precision': 0.6761822376009228, 'recall': 0.6842719579807411, 'f1-score': 0.6802030456852791, 'support': 3427}
14                   : {'precision': 0.6191819464033851, 'recall': 0.5932432432432433, 'f1-score': 0.6059351276742582, 'support': 2960}
15                   : {'precision': 0.6704048140043763, 'recall': 0.9987775061124694, 'f1-score': 0.802291325695581, 'support': 2454}
16                   : {'precision': 0.9989171629669734, 'recall': 0.9967585089141004, 'f1-score': 0.997836668469443, 'support': 1851}
17                   : {'precision': 1.0, 'recall': 0.9974337040205303, 'f1-score': 0.9987152034261242, 'support': 1169}
18                   : {'precision': 0.9983079526226735, 'recall': 0.9983079526226735, 'f1-score': 0.9983079526226735, 'support': 591}
accuracy             : 0.8636333333333334
macro avg            : {'precision': 0.8882900375101347, 'recall': 0.8949047515577803, 'f1-score': 0.888232629553482, 'support': 60000}
weighted avg         : {'precision': 0.87256034155427, 'recall': 0.8636333333333334, 'f1-score': 0.8643125189549837, 'support': 60000}
TRAIN : epoch=27 acc_l: 99.83 acc_o: 86.16 loss: 1.01434: 100%|██████████| 1875/1875 [00:40<00:00, 46.21it/s]


TEST : epoch=27 acc_l: 0.27 acc_o: 0.23 loss: 0.00283:   0%|          | 0/1875 [00:00<?, ?it/s]0                    : {'precision': 0.99915611814346, 'recall': 0.9994934999155833, 'f1-score': 0.99932478055368, 'support': 5923}
1                    : {'precision': 0.9986654804270463, 'recall': 0.9989617324236132, 'f1-score': 0.9988135844579565, 'support': 6742}
2                    : {'precision': 0.9984891724022159, 'recall': 0.9983215844243034, 'f1-score': 0.9984053713806127, 'support': 5958}
3                    : {'precision': 0.9990212071778141, 'recall': 0.998858261295058, 'f1-score': 0.9989397275915505, 'support': 6131}
4                    : {'precision': 0.9981167608286252, 'recall': 0.9979459089352961, 'f1-score': 0.9980313275699734, 'support': 5842}
5                    : {'precision': 0.9970517781463055, 'recall': 0.998155321896329, 'f1-score': 0.9976032448377582, 'support': 5421}
6                    : {'precision': 0.9988165680473373, 'recall': 0.9983102399459277, 'f1-score': 0.9985633398123892, 'support': 5918}
7                    : {'precision': 0.9979249800478851, 'recall': 0.9979249800478851, 'f1-score': 0.9979249800478851, 'support': 6265}
8                    : {'precision': 0.9979490685352931, 'recall': 0.9979490685352931, 'f1-score': 0.9979490685352931, 'support': 5851}
9                    : {'precision': 0.9971404541631623, 'recall': 0.9964699949571356, 'f1-score': 0.9968051118210862, 'support': 5949}
accuracy             : 0.99825
macro avg            : {'precision': 0.9982331587919144, 'recall': 0.9982390592376426, 'f1-score': 0.9982360536608184, 'support': 60000}
weighted avg         : {'precision': 0.9982500447465865, 'recall': 0.99825, 'f1-score': 0.9982499699224516, 'support': 60000}
0                    : {'precision': 1.0, 'recall': 1.0, 'f1-score': 1.0, 'support': 602}
1                    : {'precision': 1.0, 'recall': 0.9992254066615027, 'f1-score': 0.9996125532739248, 'support': 1291}
2                    : {'precision': 0.998992950654582, 'recall': 0.998992950654582, 'f1-score': 0.998992950654582, 'support': 1986}
3                    : {'precision': 0.9988193624557261, 'recall': 0.9988193624557261, 'f1-score': 0.9988193624557261, 'support': 2541}
4                    : {'precision': 0.8375420875420876, 'recall': 0.998995983935743, 'f1-score': 0.9111721611721612, 'support': 2988}
5                    : {'precision': 0.8398058252427184, 'recall': 0.8352740698665152, 'f1-score': 0.8375338174569272, 'support': 3521}
6                    : {'precision': 0.9988751406074241, 'recall': 0.8638132295719845, 'f1-score': 0.9264475743348982, 'support': 4112}
7                    : {'precision': 0.8884034225377754, 'recall': 0.9987720016373312, 'f1-score': 0.940360343000289, 'support': 4886}
8                    : {'precision': 0.8952254641909815, 'recall': 0.8856607310215557, 'f1-score': 0.8904174126071799, 'support': 5335}
9                    : {'precision': 0.9970154821861593, 'recall': 0.9053184281842819, 'f1-score': 0.9489569462938304, 'support': 5904}
10                   : {'precision': 0.9985915492957746, 'recall': 0.7821290678433536, 'f1-score': 0.8772038354469532, 'support': 5439}
11                   : {'precision': 0.7533305578684429, 'recall': 0.7469556243550052, 'f1-score': 0.7501295471033268, 'support': 4845}
12                   : {'precision': 0.7047158403869408, 'recall': 0.7098660170523752, 'f1-score': 0.7072815533980583, 'support': 4105}
13                   : {'precision': 0.6635514018691588, 'recall': 0.6660034110289937, 'f1-score': 0.6647751454106965, 'support': 3518}
14                   : {'precision': 0.5940730530668504, 'recall': 0.592032967032967, 'f1-score': 0.5930512555899553, 'support': 2912}
15                   : {'precision': 0.6674138414121603, 'recall': 0.9983235540653814, 'f1-score': 0.7999999999999999, 'support': 2386}
16                   : {'precision': 0.9977961432506887, 'recall': 0.9972466960352423, 'f1-score': 0.997521343982374, 'support': 1816}
17                   : {'precision': 1.0, 'recall': 0.9967078189300411, 'f1-score': 0.998351195383347, 'support': 1215}
18                   : {'precision': 0.9966666666666667, 'recall': 1.0, 'f1-score': 0.9983305509181971, 'support': 598}
accuracy             : 0.8616333333333334
macro avg            : {'precision': 0.8858325678544282, 'recall': 0.8933756484385569, 'f1-score': 0.8862609236043383, 'support': 60000}
weighted avg         : {'precision': 0.8706484332040842, 'recall': 0.8616333333333334, 'f1-score': 0.8624672001270456, 'support': 60000}
TEST : epoch=27 acc_l: 99.88 acc_o: 87.28 loss: 1.01258: 100%|██████████| 1875/1875 [00:30<00:00, 61.48it/s]


0                    : {'precision': 0.999156402901974, 'recall': 0.9998311666385278, 'f1-score': 0.999493670886076, 'support': 5923}
1                    : {'precision': 0.9985178597895361, 'recall': 0.9992583803025809, 'f1-score': 0.9988879828008007, 'support': 6742}
2                    : {'precision': 0.9996640349403662, 'recall': 0.9988251090970124, 'f1-score': 0.9992443959365291, 'support': 5958}
3                    : {'precision': 0.9990215264187867, 'recall': 0.9991844723536127, 'f1-score': 0.9991029927423958, 'support': 6131}
4                    : {'precision': 0.9984596953619715, 'recall': 0.998630605956864, 'f1-score': 0.9985451433461702, 'support': 5842}
5                    : {'precision': 0.9994457786809533, 'recall': 0.997970854085962, 'f1-score': 0.9987077718294259, 'support': 5421}
6                    : {'precision': 0.9988177672690424, 'recall': 0.999324095978371, 'f1-score': 0.9990708674719148, 'support': 5918}
7                    : {'precision': 0.998085513720485, 'recall': 0.9985634477254589, 'f1-score': 0.9983244235219022, 'support': 6265}
8                    : {'precision': 0.9986327123568621, 'recall': 0.9986327123568621, 'f1-score': 0.9986327123568621, 'support': 5851}
9                    : {'precision': 0.9986543313708999, 'recall': 0.9979828542612204, 'f1-score': 0.9983184799058349, 'support': 5949}
accuracy             : 0.9988333333333334
macro avg            : {'precision': 0.9988455622810879, 'recall': 0.9988203698756472, 'f1-score': 0.9988328440797913, 'support': 60000}
weighted avg         : {'precision': 0.9988335182959056, 'recall': 0.9988333333333334, 'f1-score': 0.9988333075788944, 'support': 60000}
0                    : {'precision': 0.9983974358974359, 'recall': 1.0, 'f1-score': 0.9991980753809142, 'support': 623}
1                    : {'precision': 0.99836867862969, 'recall': 0.99836867862969, 'f1-score': 0.99836867862969, 'support': 1226}
2                    : {'precision': 0.9977790116601888, 'recall': 0.9972253052164262, 'f1-score': 0.9975020815986677, 'support': 1802}
3                    : {'precision': 0.9996027016289233, 'recall': 0.9996027016289233, 'f1-score': 0.9996027016289233, 'support': 2517}
4                    : {'precision': 0.8347016967706623, 'recall': 0.9990173599737963, 'f1-score': 0.9094975398837036, 'support': 3053}
5                    : {'precision': 0.8411562678878076, 'recall': 0.8299915278170008, 'f1-score': 0.8355366027007817, 'support': 3541}
6                    : {'precision': 0.9991765028822399, 'recall': 0.8683206106870229, 'f1-score': 0.9291640076579452, 'support': 4192}
7                    : {'precision': 0.998365345320801, 'recall': 0.9993863775823276, 'f1-score': 0.9988756005315343, 'support': 4889}
8                    : {'precision': 0.8998849252013809, 'recall': 0.9983585628305672, 'f1-score': 0.9465675255057929, 'support': 5483}
9                    : {'precision': 0.9994385176866929, 'recall': 0.8982338099243061, 'f1-score': 0.9461374911410347, 'support': 5945}
10                   : {'precision': 0.9990379990379991, 'recall': 0.7812676321233778, 'f1-score': 0.8768337730870712, 'support': 5317}
11                   : {'precision': 0.7545764154959558, 'recall': 0.7417869847248378, 'f1-score': 0.7481270444233409, 'support': 4779}
12                   : {'precision': 0.7029750479846449, 'recall': 0.7200786434013271, 'f1-score': 0.711424062158553, 'support': 4069}
13                   : {'precision': 0.6852313660293711, 'recall': 0.6865630205441421, 'f1-score': 0.6858965469421717, 'support': 3602}
14                   : {'precision': 0.6137295081967213, 'recall': 0.6038306451612904, 'f1-score': 0.608739837398374, 'support': 2976}
15                   : {'precision': 0.6727777777777778, 'recall': 0.998351195383347, 'f1-score': 0.8038499834052439, 'support': 2426}
16                   : {'precision': 0.9994363021420518, 'recall': 0.9983108108108109, 'f1-score': 0.9988732394366198, 'support': 1776}
17                   : {'precision': 0.9991554054054054, 'recall': 1.0, 'f1-score': 0.9995775242923532, 'support': 1183}
18                   : {'precision': 0.9983388704318937, 'recall': 1.0, 'f1-score': 0.9991687448046551, 'support': 601}
accuracy             : 0.8727833333333334
macro avg            : {'precision': 0.8943226197930338, 'recall': 0.9009838877073261, 'f1-score': 0.8943653189793354, 'support': 60000}
weighted avg         : {'precision': 0.8815733162854656, 'recall': 0.8727833333333334, 'f1-score': 0.8735043868911591, 'support': 60000}
TRAIN : epoch=28 acc_l: 99.83 acc_o: 87.28 loss: 1.01442: 100%|██████████| 1875/1875 [00:40<00:00, 46.25it/s]


TEST : epoch=28 acc_l: 0.27 acc_o: 0.23 loss: 0.00251:   0%|          | 0/1875 [00:00<?, ?it/s]0                    : {'precision': 0.99915611814346, 'recall': 0.9994934999155833, 'f1-score': 0.99932478055368, 'support': 5923}
1                    : {'precision': 0.9982211680996146, 'recall': 0.9988134084841294, 'f1-score': 0.9985172004744959, 'support': 6742}
2                    : {'precision': 0.9991602284178703, 'recall': 0.9984894259818731, 'f1-score': 0.9988247145735393, 'support': 5958}
3                    : {'precision': 0.9991839399379794, 'recall': 0.998532050236503, 'f1-score': 0.9988578887257302, 'support': 6131}
4                    : {'precision': 0.9986289631533848, 'recall': 0.9974323861691201, 'f1-score': 0.9980303160058234, 'support': 5842}
5                    : {'precision': 0.9977863862755949, 'recall': 0.9977863862755949, 'f1-score': 0.9977863862755949, 'support': 5421}
6                    : {'precision': 0.9984802431610942, 'recall': 0.9991551199729638, 'f1-score': 0.9988175675675675, 'support': 5918}
7                    : {'precision': 0.9974477588132078, 'recall': 0.9980845969672786, 'f1-score': 0.9977660762725388, 'support': 6265}
8                    : {'precision': 0.9981203007518797, 'recall': 0.9982908904460776, 'f1-score': 0.9982055883106896, 'support': 5851}
9                    : {'precision': 0.9971423768700622, 'recall': 0.9971423768700622, 'f1-score': 0.9971423768700622, 'support': 5949}
accuracy             : 0.9983333333333333
macro avg            : {'precision': 0.9983327483624148, 'recall': 0.9983220141319185, 'f1-score': 0.9983272895629722, 'support': 60000}
weighted avg         : {'precision': 0.9983335125033026, 'recall': 0.9983333333333333, 'f1-score': 0.9983333307008213, 'support': 60000}
0                    : {'precision': 1.0, 'recall': 0.9982935153583617, 'f1-score': 0.9991460290350128, 'support': 586}
1                    : {'precision': 1.0, 'recall': 1.0, 'f1-score': 1.0, 'support': 1254}
2                    : {'precision': 0.9994481236203091, 'recall': 0.9988968560397132, 'f1-score': 0.9991724137931035, 'support': 1813}
3                    : {'precision': 0.9984031936127744, 'recall': 0.9992009588493808, 'f1-score': 0.9988019169329073, 'support': 2503}
4                    : {'precision': 0.8421489891996676, 'recall': 0.9990144546649146, 'f1-score': 0.913899323816679, 'support': 3044}
5                    : {'precision': 0.8452121871599565, 'recall': 0.8452121871599565, 'f1-score': 0.8452121871599565, 'support': 3676}
6                    : {'precision': 0.9778796870785001, 'recall': 0.8637121753633548, 'f1-score': 0.9172570850202428, 'support': 4197}
7                    : {'precision': 0.9959250203748982, 'recall': 0.9835010060362173, 'f1-score': 0.9896740230815955, 'support': 4970}
8                    : {'precision': 0.9053004726536125, 'recall': 0.9966548968593198, 'f1-score': 0.9487837240159223, 'support': 5381}
9                    : {'precision': 0.999062089664228, 'recall': 0.9051665533650578, 'f1-score': 0.949799375835934, 'support': 5884}
10                   : {'precision': 0.99853515625, 'recall': 0.7722809667673716, 'f1-score': 0.870954003407155, 'support': 5296}
11                   : {'precision': 0.7452810180275716, 'recall': 0.7463891248937978, 'f1-score': 0.7458346598747745, 'support': 4708}
12                   : {'precision': 0.717417202763879, 'recall': 0.723798076923077, 'f1-score': 0.72059351441905, 'support': 4160}
13                   : {'precision': 0.6786214625945644, 'recall': 0.6888509670079636, 'f1-score': 0.683697953422724, 'support': 3516}
14                   : {'precision': 0.6133004926108374, 'recall': 0.5985576923076923, 'f1-score': 0.6058394160583941, 'support': 2912}
15                   : {'precision': 0.6787280701754386, 'recall': 0.9983870967741936, 'f1-score': 0.8080939947780679, 'support': 2480}
16                   : {'precision': 0.9955727725511898, 'recall': 0.9966759002770084, 'f1-score': 0.996124031007752, 'support': 1805}
17                   : {'precision': 0.9991701244813278, 'recall': 0.9983416252072969, 'f1-score': 0.9987557030277893, 'support': 1206}
18                   : {'precision': 0.9983552631578947, 'recall': 0.9967159277504105, 'f1-score': 0.9975349219391947, 'support': 609}
accuracy             : 0.8727833333333334
macro avg            : {'precision': 0.8941242803145606, 'recall': 0.9005078937686889, 'f1-score': 0.8941670671908557, 'support': 60000}
weighted avg         : {'precision': 0.8812002102739952, 'recall': 0.8727833333333334, 'f1-score': 0.8734355839790277, 'support': 60000}
TEST : epoch=28 acc_l: 99.89 acc_o: 86.25 loss: 1.01033: 100%|██████████| 1875/1875 [00:30<00:00, 61.18it/s]


0                    : {'precision': 0.999493670886076, 'recall': 0.9998311666385278, 'f1-score': 0.99966239027684, 'support': 5923}
1                    : {'precision': 0.9985180794309425, 'recall': 0.9994067042420647, 'f1-score': 0.9989621942179392, 'support': 6742}
2                    : {'precision': 0.999664147774979, 'recall': 0.9991607922121517, 'f1-score': 0.9994124066146227, 'support': 5958}
3                    : {'precision': 0.9990216859611936, 'recall': 0.9993475778828902, 'f1-score': 0.9991846053489889, 'support': 6131}
4                    : {'precision': 0.998289136013687, 'recall': 0.9988017802122561, 'f1-score': 0.9985453923162488, 'support': 5842}
5                    : {'precision': 0.9987082487543827, 'recall': 0.9983397897066962, 'f1-score': 0.9985239852398524, 'support': 5421}
6                    : {'precision': 0.9989868287740629, 'recall': 0.9996620479891856, 'f1-score': 0.9993243243243244, 'support': 5918}
7                    : {'precision': 0.9985634477254589, 'recall': 0.9985634477254589, 'f1-score': 0.9985634477254589, 'support': 6265}
8                    : {'precision': 0.9991445680068435, 'recall': 0.9981199794906853, 'f1-score': 0.9986320109439124, 'support': 5851}
9                    : {'precision': 0.9988221436984688, 'recall': 0.9978147587829888, 'f1-score': 0.998318197107299, 'support': 5949}
accuracy             : 0.9989166666666667
macro avg            : {'precision': 0.9989211957026095, 'recall': 0.9989048044882904, 'f1-score': 0.9989128954115486, 'support': 60000}
weighted avg         : {'precision': 0.998916754254961, 'recall': 0.9989166666666667, 'f1-score': 0.9989166048817317, 'support': 60000}
0                    : {'precision': 0.998220640569395, 'recall': 0.998220640569395, 'f1-score': 0.998220640569395, 'support': 562}
1                    : {'precision': 1.0, 'recall': 1.0, 'f1-score': 1.0, 'support': 1264}
2                    : {'precision': 0.9984051036682615, 'recall': 1.0, 'f1-score': 0.9992019154030327, 'support': 1878}
3                    : {'precision': 0.9992012779552716, 'recall': 1.0, 'f1-score': 0.9996004794246904, 'support': 2502}
4                    : {'precision': 0.8292282430213465, 'recall': 0.9993403693931399, 'f1-score': 0.9063715225845048, 'support': 3032}
5                    : {'precision': 0.8302425106990015, 'recall': 0.8229638009049773, 'f1-score': 0.8265871325095866, 'support': 3536}
6                    : {'precision': 0.9989082969432315, 'recall': 0.8599624060150376, 'f1-score': 0.9242424242424243, 'support': 4256}
7                    : {'precision': 0.8872468527640941, 'recall': 0.9997944078947368, 'f1-score': 0.94016433059449, 'support': 4864}
8                    : {'precision': 0.8954579579579579, 'recall': 0.8851576994434137, 'f1-score': 0.8902780369471917, 'support': 5390}
9                    : {'precision': 0.998894620486367, 'recall': 0.9066889632107024, 'f1-score': 0.9505610098176719, 'support': 5980}
10                   : {'precision': 0.9983552631578947, 'recall': 0.7865605331358756, 'f1-score': 0.8798923172499482, 'support': 5402}
11                   : {'precision': 0.7558287409919457, 'recall': 0.7433812799666458, 'f1-score': 0.7495533368365739, 'support': 4797}
12                   : {'precision': 0.7049533381191673, 'recall': 0.7117661270838367, 'f1-score': 0.7083433517672518, 'support': 4139}
13                   : {'precision': 0.667226420375035, 'recall': 0.6807538549400343, 'f1-score': 0.6739222614840991, 'support': 3502}
14                   : {'precision': 0.611731843575419, 'recall': 0.6049723756906077, 'f1-score': 0.6083333333333333, 'support': 2896}
15                   : {'precision': 0.6681184668989547, 'recall': 0.9991315675206253, 'f1-score': 0.8007656168435706, 'support': 2303}
16                   : {'precision': 0.9994750656167979, 'recall': 0.9984268484530676, 'f1-score': 0.9989506820566632, 'support': 1907}
17                   : {'precision': 0.9982993197278912, 'recall': 0.9982993197278912, 'f1-score': 0.9982993197278912, 'support': 1176}
18                   : {'precision': 0.998371335504886, 'recall': 0.998371335504886, 'f1-score': 0.998371335504886, 'support': 614}
accuracy             : 0.8625
macro avg            : {'precision': 0.8862192262122589, 'recall': 0.8944100804976248, 'f1-score': 0.8869294235209054, 'support': 60000}
weighted avg         : {'precision': 0.8715550025247024, 'recall': 0.8625, 'f1-score': 0.863358511999956, 'support': 60000}
TRAIN : epoch=29 acc_l: 99.83 acc_o: 86.47 loss: 1.01598: 100%|██████████| 1875/1875 [00:41<00:00, 45.05it/s]


TEST : epoch=29 acc_l: 0.27 acc_o: 0.23 loss: 0.00285:   0%|          | 0/1875 [00:00<?, ?it/s]0                    : {'precision': 0.9993247805536799, 'recall': 0.9994934999155833, 'f1-score': 0.9994091331138685, 'support': 5923}
1                    : {'precision': 0.9985167606051617, 'recall': 0.9985167606051617, 'f1-score': 0.9985167606051617, 'support': 6742}
2                    : {'precision': 0.9988247145735393, 'recall': 0.9984894259818731, 'f1-score': 0.9986570421353029, 'support': 5958}
3                    : {'precision': 0.9988586336213925, 'recall': 0.9991844723536127, 'f1-score': 0.9990215264187867, 'support': 6131}
4                    : {'precision': 0.9974332648870636, 'recall': 0.9977747346799042, 'f1-score': 0.9976039705630669, 'support': 5842}
5                    : {'precision': 0.9985226223453371, 'recall': 0.9974174506548608, 'f1-score': 0.9979697305278701, 'support': 5421}
6                    : {'precision': 0.9981428330238055, 'recall': 0.9989861439675566, 'f1-score': 0.9985643104467528, 'support': 5918}
7                    : {'precision': 0.9979249800478851, 'recall': 0.9979249800478851, 'f1-score': 0.9979249800478851, 'support': 6265}
8                    : {'precision': 0.9976080642405604, 'recall': 0.9979490685352931, 'f1-score': 0.9977785372522214, 'support': 5851}
9                    : {'precision': 0.9973086627417999, 'recall': 0.9966380904353673, 'f1-score': 0.9969732638305029, 'support': 5949}
accuracy             : 0.99825
macro avg            : {'precision': 0.9982465316640224, 'recall': 0.9982374627177097, 'f1-score': 0.9982419254941419, 'support': 60000}
weighted avg         : {'precision': 0.9982500030157169, 'recall': 0.99825, 'f1-score': 0.9982499332230745, 'support': 60000}
0                    : {'precision': 0.9982486865148862, 'recall': 1.0, 'f1-score': 0.9991235758106924, 'support': 570}
1                    : {'precision': 1.0, 'recall': 0.9984387197501952, 'f1-score': 0.99921875, 'support': 1281}
2                    : {'precision': 0.9983818770226537, 'recall': 0.997843665768194, 'f1-score': 0.9981126988406578, 'support': 1855}
3                    : {'precision': 0.9987674609695973, 'recall': 0.9991779695848746, 'f1-score': 0.9989726731045818, 'support': 2433}
4                    : {'precision': 0.8445105462412115, 'recall': 0.9990403071017274, 'f1-score': 0.9152989449003517, 'support': 3126}
5                    : {'precision': 0.8458710407239819, 'recall': 0.8380498739142617, 'f1-score': 0.841942294159043, 'support': 3569}
6                    : {'precision': 0.998099891422367, 'recall': 0.8713270142180095, 'f1-score': 0.9304149797570851, 'support': 4220}
7                    : {'precision': 0.9030877976190477, 'recall': 0.9987656860728246, 'f1-score': 0.9485200742405002, 'support': 4861}
8                    : {'precision': 0.8926811461945611, 'recall': 0.9038994640547033, 'f1-score': 0.8982552800734619, 'support': 5411}
9                    : {'precision': 0.9983443708609272, 'recall': 0.9014950166112957, 'f1-score': 0.9474511173184358, 'support': 6020}
10                   : {'precision': 0.9971489665003563, 'recall': 0.7790978281046965, 'f1-score': 0.8747394747811588, 'support': 5387}
11                   : {'precision': 0.7504729871767921, 'recall': 0.7490558120016786, 'f1-score': 0.749763729917043, 'support': 4766}
12                   : {'precision': 0.7125662012518055, 'recall': 0.7267370488583353, 'f1-score': 0.7195818645921964, 'support': 4073}
13                   : {'precision': 0.6861747243426632, 'recall': 0.6741666666666667, 'f1-score': 0.6801176965111392, 'support': 3600}
14                   : {'precision': 0.5939015939015939, 'recall': 0.5912383580545015, 'f1-score': 0.5925669835782195, 'support': 2899}
15                   : {'precision': 0.6619196810025634, 'recall': 0.9974248927038627, 'f1-score': 0.795754151686355, 'support': 2330}
16                   : {'precision': 0.9966235227912211, 'recall': 0.9983089064261556, 'f1-score': 0.9974655026753028, 'support': 1774}
17                   : {'precision': 1.0, 'recall': 0.9991708126036484, 'f1-score': 0.9995852343425965, 'support': 1206}
18                   : {'precision': 1.0, 'recall': 0.9967689822294022, 'f1-score': 0.9983818770226538, 'support': 619}
accuracy             : 0.8646666666666667
macro avg            : {'precision': 0.88825265760717, 'recall': 0.8957898434065806, 'f1-score': 0.888698258069025, 'support': 60000}
weighted avg         : {'precision': 0.8735164631656018, 'recall': 0.8646666666666667, 'f1-score': 0.8654779180532839, 'support': 60000}
TEST : epoch=29 acc_l: 99.89 acc_o: 86.07 loss: 1.01113: 100%|██████████| 1875/1875 [00:31<00:00, 60.34it/s]


0                    : {'precision': 0.99949358541526, 'recall': 0.9996623332770556, 'f1-score': 0.9995779522241918, 'support': 5923}
1                    : {'precision': 0.9986660738105825, 'recall': 0.9994067042420647, 'f1-score': 0.9990362517606938, 'support': 6742}
2                    : {'precision': 0.9996640349403662, 'recall': 0.9988251090970124, 'f1-score': 0.9992443959365291, 'support': 5958}
3                    : {'precision': 0.9990216859611936, 'recall': 0.9993475778828902, 'f1-score': 0.9991846053489889, 'support': 6131}
4                    : {'precision': 0.9982888432580425, 'recall': 0.998630605956864, 'f1-score': 0.9984596953619715, 'support': 5842}
5                    : {'precision': 0.9988927846466138, 'recall': 0.9985242575170633, 'f1-score': 0.9987084870848709, 'support': 5421}
6                    : {'precision': 0.9988181664696945, 'recall': 0.9996620479891856, 'f1-score': 0.9992399290600457, 'support': 5918}
7                    : {'precision': 0.9987224528904504, 'recall': 0.998244213886672, 'f1-score': 0.998483276123573, 'support': 6265}
8                    : {'precision': 0.9988028048571918, 'recall': 0.9981199794906853, 'f1-score': 0.9984612754316978, 'support': 5851}
9                    : {'precision': 0.9983187626092804, 'recall': 0.998150949739452, 'f1-score': 0.9982348491216273, 'support': 5949}
accuracy             : 0.9988666666666667
macro avg            : {'precision': 0.9988689194858675, 'recall': 0.9988573779078944, 'f1-score': 0.9988630717454189, 'support': 60000}
weighted avg         : {'precision': 0.9988667377873242, 'recall': 0.9988666666666667, 'f1-score': 0.9988666243439608, 'support': 60000}
0                    : {'precision': 0.9983498349834984, 'recall': 1.0, 'f1-score': 0.9991742361684558, 'support': 605}
1                    : {'precision': 1.0, 'recall': 1.0, 'f1-score': 1.0, 'support': 1258}
2                    : {'precision': 0.9989224137931034, 'recall': 0.9994609164420485, 'f1-score': 0.9991915925626516, 'support': 1855}
3                    : {'precision': 0.9991873222267371, 'recall': 0.998375964271214, 'f1-score': 0.9987814784727864, 'support': 2463}
4                    : {'precision': 0.8294197031039137, 'recall': 0.9996746909564086, 'f1-score': 0.9066233957810887, 'support': 3074}
5                    : {'precision': 0.841643059490085, 'recall': 0.824361820199778, 'f1-score': 0.8329128118867396, 'support': 3604}
6                    : {'precision': 0.9991652754590985, 'recall': 0.8659271762720039, 'f1-score': 0.9277871076088361, 'support': 4147}
7                    : {'precision': 0.8804659498207885, 'recall': 0.9993897477624084, 'f1-score': 0.9361661585365852, 'support': 4916}
8                    : {'precision': 0.8886198547215496, 'recall': 0.8770220588235295, 'f1-score': 0.8827828661300767, 'support': 5440}
9                    : {'precision': 0.9981556621172999, 'recall': 0.8997506234413966, 'f1-score': 0.9464020285039783, 'support': 6015}
10                   : {'precision': 0.9990723562152134, 'recall': 0.782703488372093, 'f1-score': 0.8777506112469438, 'support': 5504}
11                   : {'precision': 0.7433533447684391, 'recall': 0.7416042780748663, 'f1-score': 0.7424777813470391, 'support': 4675}
12                   : {'precision': 0.7088363109608885, 'recall': 0.7203140333660452, 'f1-score': 0.7145290825018255, 'support': 4076}
13                   : {'precision': 0.6741250717154331, 'recall': 0.6821480406386067, 'f1-score': 0.6781128264319723, 'support': 3445}
14                   : {'precision': 0.6113676731793961, 'recall': 0.5816154106116931, 'f1-score': 0.596120540353308, 'support': 2959}
15                   : {'precision': 0.6614796614796615, 'recall': 0.9983518747424804, 'f1-score': 0.7957307060755338, 'support': 2427}
16                   : {'precision': 0.9988998899889989, 'recall': 0.9994496422674739, 'f1-score': 0.9991746905089409, 'support': 1817}
17                   : {'precision': 0.9974048442906575, 'recall': 0.9982683982683983, 'f1-score': 0.9978364344439637, 'support': 1155}
18                   : {'precision': 1.0, 'recall': 0.9964601769911504, 'f1-score': 0.9982269503546098, 'support': 565}
accuracy             : 0.8607333333333334
macro avg            : {'precision': 0.8857088541218295, 'recall': 0.8928883337632418, 'f1-score': 0.8857779631008069, 'support': 60000}
weighted avg         : {'precision': 0.8700154461609995, 'recall': 0.8607333333333334, 'f1-score': 0.8614558616404755, 'support': 60000}
```
