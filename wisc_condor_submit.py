import os
import sys
import pwd


def retryLogic(command):
    return '''\nn=0
until [ $n -ge 5 ]
do
\techo "attempting copy for the ${{n}} time"
\t{} && break
\tn=$[$n+1]
done
'''.format(command)


def main(args):
    print "Begin NN Classifying..."
    jobName = args.jobName
    sampledir = args.sampledir
    print 'Processing samples from {} in job {}'.format(sampledir, jobName)

    head_dir = '/nfs_scratch/{}/{}'.format(pwd.getpwuid(os.getuid())[0], jobName)

    if os.path.exists(head_dir):
        print 'Submission directory exists for {}.'.format(jobName)
        return

    exe_dir = '{}/executables'.format(head_dir)
    os.system('mkdir -p {}'.format(exe_dir))

    os.system('cp condor_classify.py ${CMSSW_BASE}/bin/${SCRAM_ARCH}')
    os.system('cp {} {}'.format(args.model_vbf, head_dir))
    os.system('cp {} {}'.format(args.input_vbf, head_dir))

    model_vbf = args.model_vbf.replace('models/', '')
    input_vbf = args.input_vbf.replace('datasets/', '')

    if args.location == 'wisc':
      extension = '/cms-lvs-gridftp.hep.wisc.edu/'
    elif args.location == 'lpc':
      extension = '/cmseos-gridftp.fnal.gov/'

    fileList = [ifile for ifile in filter(None, os.popen(
        'gfal-ls gsiftp:/{}/{}'.format(extension, sampledir)).read().split('\n')) if '.root' in ifile]

    config_name = '{}/config.jdl'.format(head_dir)
    condorConfig = '''universe = vanilla
Executable = {}/executables/NN_overseer.sh
Should_Transfer_Files = YES
WhenToTransferOutput = ON_EXIT
Output = logs/{}_$(Cluster)_$(Process).stdout
Error = logs/{}_$(Cluster)_$(Process).stderr
x509userproxy = $ENV(X509_USER_PROXY)
Arguments=$(process)
Queue {}
    '''.format(head_dir, jobName, jobName, len(fileList))
    with open(config_name, 'w') as file:
        file.write(condorConfig)

    print 'Condor config has been written: {}'.format(config_name)

    NN_overseer_name = '{}/NN_overseer.sh'.format(exe_dir)
    overseerScript = '''#!/bin/bash
let "sample=${{1}}+1"
echo $sample
cp -r {} .
cd executables
echo `ls`
bash {}_${{sample}}.sh
    '''.format(exe_dir, jobName)
    with open(NN_overseer_name, 'w') as file:
        file.write(overseerScript)

    print 'Condor NN_overseer has been written: {}'.format(NN_overseer_name)

    bashScriptSetup = '''#!/bin/bash
source /cvmfs/cms.cern.ch/cmsset_default.sh
export SCRAM_ARCH=slc7_amd64_gcc700
eval `scramv1 project CMSSW CMSSW_10_4_0`
cd CMSSW_10_4_0/src
eval `scramv1 runtime -sh`
cp {}/{} .
cp {}/{} .
echo `ls`'''.format(head_dir, args.model_vbf, head_dir, args.input_vbf)

    i = 1
    for ifile in fileList:
        input_file =  ifile
        output_file = input_file.replace('.root', '_NNed.root')
        copycommand = 'cp {} .'.format(input_file)

        # create the bash config script
        bashScript = bashScriptSetup + retryLogic(copycommand)
        bash_name = '{}/{}_{}.sh'.format(exe_dir, jobName, i)

        command = 'python ${{CMSSW_BASE}}/bin/${{SCRAM_ARCH}}/condor_classify.py -t {} -o {} -f {} '.format(args.channel, output_file, input_file)
        if args.model_vbf != None and args.input_vbf != None:
            command += '--model-vbf {} --input-vbf {}'.format(
                model_vbf, input_vbf)
        if args.model_boost != None and args.input_boost != None:
            command += '--model-boost {} --input-boost {}'.format(
                args.model_boost, args.input_boost)

        bashScript += command
        bashScript += '\n'

        with open(bash_name, 'w') as file:
            file.write(bashScript)
        os.system('chmod +x {}'.format(bash_name))
        i += 1

    print 'All executables have been written.'

    if not args.dryrun:
        print 'Now submitting to condor...'
        os.system('condor_submit {}'.format(config_name))

    return


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-dr', '--dryrun', action='store_true', help='Create jobs but dont submit')
    parser.add_argument('-c', '--channel', action='store', help='name of channel [etau, mutau]')
    parser.add_argument('--model-vbf', action='store', dest='model_vbf', default=None, help='name of model to use')
    parser.add_argument('--model-boost', action='store', dest='model_boost', default=None, help='name of model to use')
    parser.add_argument('--input-vbf', action='store', dest='input_vbf', default=None, help='name of input dataset')
    parser.add_argument('--input-boost', action='store', dest='input_boost', default=None, help='name of input dataset')
    parser.add_argument('-jn', '--jobName', nargs='?', type=str, const='', help='Job Name for condor submission')
    parser.add_argument('-sd', '--sampledir', nargs='?', type=str, const='', help='The Sample Input directory')
    args = parser.parse_args()
    main(args)
