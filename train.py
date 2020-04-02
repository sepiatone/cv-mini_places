import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models.resnet import resnet18, resnet34, resnet50
from torch.autograd import Variable
import time
import argparse

from data_utils import Miniplaces

print("Torchvision and DataUtils loaded!")

#def to_var(x):
#	if torch.cuda.is_available():
#		x = x.cuda()
#	return Variable(x)
#
#def main(args):
#	data_path = args.data_path
#
#	transform = transforms.Compose([
#			transforms.ToTensor(),
#			transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
#		])
#	miniplaces = Miniplaces(data_path, 'train', transform)
#	places_loader = DataLoader(miniplaces, batch_size = args.batch_size, shuffle=True)
#
#	if args.checkpoint != None:
#		checkpoint = torch.load(args.checkpoint)
#		model_type = checkpoint['args']['model']
#	else:
#		model_type = args.model
#
#	if model_type == 'resnet34':
#		model = resnet34(num_classes=100)
#	elif model_type == 'resnet50':
#		model = resnet50(num_classes=100)
#	else:
#		model_type = 'resnet18'
#		model = resnet18(num_classes=100)
#	model.avgpool = torch.nn.AdaptiveAvgPool2d(1)
#
#	if args.checkpoint != None:
#		model.load_state_dict(checkpoint['model_state'])
#		count = checkpoint['iter']
#	else:
#		count = 0
#
#	criterion = torch.nn.CrossEntropyLoss()
#	optimizer = torch.optim.Adam(params=model.parameters(), lr=args.learning_rate)
#
#	if torch.cuda.is_available():
#		print('Training %s with %d GPUs' % (model_type, torch.cuda.device_count()))
#		model = model.cuda()
#		criterion = criterion.cuda()
#	else:
#		print('Training %s with CPU' % model_type)
#
#	model.train()
#	clock = time.time()
#	count_0 = count
#	for epoch in range(args.num_epochs):
#		for img_batch, label_batch in places_loader:
#
#			img_batch = to_var(img_batch)
#			label_batch = to_var(label_batch)
#
#			logits = model(img_batch)
#			loss = criterion(logits, label_batch)
#
#			optimizer.zero_grad()
#			loss.backward()
#			optimizer.step()
#
#			count += 1
#			time_ellapsed = time.time() - clock
#
#			if count % args.show_every == 0:
#				print('Running epoch %d. Total iteration %d. Loss = %.4f. Estimated time remaining = %ds'
#				 % (epoch+1, count, loss.data[0], int(time_ellapsed * (len(miniplaces) / args.batch_size * args.num_epochs - count + count_0) / (count - count_0))))
#			if count % args.save_every == 0:
#				print('Saving checkpoint')
#				checkpoint = {
#					'args': args.__dict__,
#					'iter': count,
#					'model_state': model.state_dict()
#				}
#				torch.save(checkpoint, args.result_path + 'checkpoint.pt')
#	
#	result = {
#		'args': args.__dict__,
#		'iter': count,
#		'model_state': model.state_dict()
#	}
#	print('Finished training. Total iteration = %d. Saving result to %s'
#		% (count, args.result_path + 'result.pt'))
#	torch.save(result, args.result_path + 'result.pt')
#
#if __name__ == '__main__':
#	parser = argparse.ArgumentParser()
#	parser.add_argument('--checkpoint', type=str, default=None,
#                        help='start from previous checkpoint')
#	parser.add_argument('--model', type=str, default='resnet18',
#                        help='network architecture')
#	parser.add_argument('--batch_size', type=int, default=100,
#                        help='batch size for training')
#	parser.add_argument('--learning_rate', type=float, default=1e-3,
#                        help='learning rate for adam optimizer')
#	parser.add_argument('--num_epochs', type=int, default=5,
#                        help='number of training epochs')
#	parser.add_argument('--save_every', type=int, default=200,
#                        help='save model every ? iteration')
#	parser.add_argument('--show_every', type=int, default=20,
#                        help='display log info every ? iteration')
#	parser.add_argument('--data_path', type=str, default='data/',
#                        help='data directory')
#	parser.add_argument('--result_path', type=str, default='data/',
#                        help='save result and checkpoint to')
#	args = parser.parse_args()
#main(args)
#
