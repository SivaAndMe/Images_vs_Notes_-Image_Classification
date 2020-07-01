
mod = torch.load('/content/drive/My Drive/ML/saved_models/imgsvsnotes_transfer_learning_deepnet')
# mod.eval()
correc=0
with torch.no_grad():# for saving memory dont compute grads while testing
    mod.eval()
    for xv,yv in test_dataloader:
      predv = mod(xv)
      predv = (predv>0.5).float()
      yv = yv.reshape(shape=(yv.shape[0],1))
      correc += (predv==yv).int().sum()

    print(" Testing: Correct:{}  Accuracy : {}".format(correc,(1.0*correc)/len(testdata)))
