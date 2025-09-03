**test_1** | Caltech101 | sgd(lr=0.01, mom=0.9, wd=0.0, epochs=300, cosine) | fp16 | GP(matern, k=128, weight=sigmoid, mc=20, lr=0.1, β=1.0) | 7 templates | batch=128 | transforms=[crop, flip, normalize]
**test_2** | Caltech101 | sgd(lr=0.01, mom=0.9, wd=0.0, epochs=300, cosine) | fp16 | GP(rbf, k=64, weight=sigmoid, mc=20, lr=0.1, β=1.0) | 7 templates | batch=128 | transforms=[crop, flip, normalize]
**test_3** — k=64, PREC=fp16, GP(kernel=rbf, mc=30, lr=0.01, β=1)+linear mean, SGD(lr=0.01, cosine, 300 epochs, momentum=0.9), batch=128x128, 7 templates
