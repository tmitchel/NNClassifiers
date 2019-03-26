import os
import sys
import pwd


def main(args):
    '''
    Submit a job using farmoutAnalysisJobs --fwklite
    '''
    print "Begin NN classifying with Condor"
    jobName = args.jobName
    sampledir = args.sampledir
    sample_name = os.path.basename(sampledir)
    print "sample_name:", args.samplename
    if sample_name == '':
        print "SAMPLE_NAME not defined, check for trailing '/' on sampledir path"
        return
    else:
        sample_dir = '/nfs_scratch/%s/%s/%s' % (
            pwd.getpwuid(os.getuid())[0], jobName, args.samplename)

    # create submit dir
    submit_dir = '%s/submit' % (sample_dir)
    if os.path.exists(submit_dir):
        print('Submission directory exists for %s %s.' %
              (jobName, args.samplename))

    # create dag dir
    dag_dir = '%s/dags/dag' % (sample_dir)
    os.system('mkdir -p %s' % (os.path.dirname(dag_dir)))
    os.system('mkdir -p %s' % (dag_dir+'inputs'))

    # output dir
    output_dir = 'gsiftp://cms-lvs-gridftp.hep.wisc.edu:2811//hdfs/store/user/%s/%s/'\
        % (pwd.getpwuid(os.getuid())[0], jobName)

    # create file list
    filelist = ['%s/%s' % (sampledir, x) for x in os.listdir(sampledir)]
    filesperjob = 1
    input_name = '%s/%s.txt' % (dag_dir+'inputs', args.samplename)
    with open(input_name, 'w') as file:
        for f in filelist:
            file.write('%s\n' % f.replace('/hdfs', '', 1))

    # create bash script
    bash_name = '%s/%s.sh' % (dag_dir+'inputs', args.samplename)
    bashScript = "#!/bin/bash\n value=$(<$INPUT)\n echo \"$value\"\n"


    command = 'python condor_classify.py -t {} -o output_files -f $value '.format(args.channel)
    if args.model_vbf != None and args.input_vbf != None:
        command += '--model-vbf {} --input-vbf {}'.format(args.model_vbf, args.input_vbf)
    if args.model_boost != None and args.input_boost != None:
        command += '--model-boost {} --input-boost {}'.format(args.model_boost, args.input_boost)
    bashScript += command

    bashScript += '\n'
    with open(bash_name, 'w') as file:
        file.write(bashScript)
    os.system('chmod +x %s' % bash_name)

    # create farmout command
    farmoutString = 'farmoutAnalysisJobs --infer-cmssw-path --fwklite --input-file-list=%s' % (
        input_name)
    farmoutString += ' --submit-dir=%s --output-dag-file=%s --output-dir=%s' % (
        submit_dir, dag_dir, output_dir)
    farmoutString += ' --input-files-per-job=%i %s %s ' % (
        filesperjob, jobName, bash_name)
    farmoutString += '--use-hdfs'

    if not args.dryrun:
        print('Submitting %s' % args.samplename)
        os.system(farmoutString)
    else:
        print farmoutString

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
    parser.add_argument('-sn', '--samplename', nargs='?', type=str, const='', help='Name of samples')
    parser.add_argument('-sd', '--sampledir', nargs='?', type=str, const='', help='The Sample Input directory')
    args = parser.parse_args()
    main(args)
