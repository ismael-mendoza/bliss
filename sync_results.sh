rsync -avL --progress -e 'ssh -i ../../bryans_key_oregon.pem' \
   ubuntu@ec2-50-112-197-133.us-west-2.compute.amazonaws.com:/home/ubuntu/astronomy/DeblendingStarfields/fits/.\
   /home/runjing_liu/Documents/astronomy/DeblendingStarfields/fits/.

