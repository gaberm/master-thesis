{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/var/folders/k9/7_28zhp52b3f5wvy8fcyp8440000gn/T/ipykernel_9006/1060207763.py:3: UserWarning: \n",
      "The version_base parameter is not specified.\n",
      "Please specify a compatability version level, or None.\n",
      "Will assume defaults for version 1.1\n",
      "  @hydra.main(config_path=\"conf\", config_name=\"config\")\n",
      "usage: ipykernel_launcher.py [--help] [--hydra-help] [--version]\n",
      "                             [--cfg {job,hydra,all}] [--resolve]\n",
      "                             [--package PACKAGE] [--run] [--multirun]\n",
      "                             [--shell-completion] [--config-path CONFIG_PATH]\n",
      "                             [--config-name CONFIG_NAME]\n",
      "                             [--config-dir CONFIG_DIR]\n",
      "                             [--experimental-rerun EXPERIMENTAL_RERUN]\n",
      "                             [--info [{all,config,defaults,defaults-tree,plugins,searchpath}]]\n",
      "                             [overrides ...]\n",
      "ipykernel_launcher.py: error: argument --shell-completion/-sc: ignored explicit argument '9002'\n",
      "ERROR:root:Internal Python error in the inspect module.\n",
      "Below is the traceback from this internal error.\n",
      "\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Traceback (most recent call last):\n",
      "  File \"/opt/anaconda3/lib/python3.9/argparse.py\", line 1858, in parse_known_args\n",
      "    namespace, args = self._parse_known_args(args, namespace)\n",
      "  File \"/opt/anaconda3/lib/python3.9/argparse.py\", line 2067, in _parse_known_args\n",
      "    start_index = consume_optional(start_index)\n",
      "  File \"/opt/anaconda3/lib/python3.9/argparse.py\", line 1989, in consume_optional\n",
      "    raise ArgumentError(action, msg % explicit_arg)\n",
      "argparse.ArgumentError: argument --shell-completion/-sc: ignored explicit argument '9002'\n",
      "\n",
      "During handling of the above exception, another exception occurred:\n",
      "\n",
      "Traceback (most recent call last):\n",
      "  File \"/opt/anaconda3/lib/python3.9/site-packages/IPython/core/interactiveshell.py\", line 3457, in run_code\n",
      "    exec(code_obj, self.user_global_ns, self.user_ns)\n",
      "  File \"/var/folders/k9/7_28zhp52b3f5wvy8fcyp8440000gn/T/ipykernel_9006/1060207763.py\", line 9, in <module>\n",
      "    main()\n",
      "  File \"/opt/anaconda3/lib/python3.9/site-packages/hydra/main.py\", line 86, in decorated_main\n",
      "    args = args_parser.parse_args()\n",
      "  File \"/opt/anaconda3/lib/python3.9/argparse.py\", line 1825, in parse_args\n",
      "    args, argv = self.parse_known_args(args, namespace)\n",
      "  File \"/opt/anaconda3/lib/python3.9/argparse.py\", line 1861, in parse_known_args\n",
      "    self.error(str(err))\n",
      "  File \"/opt/anaconda3/lib/python3.9/argparse.py\", line 2582, in error\n",
      "    self.exit(2, _('%(prog)s: error: %(message)s\\n') % args)\n",
      "  File \"/opt/anaconda3/lib/python3.9/argparse.py\", line 2569, in exit\n",
      "    _sys.exit(status)\n",
      "SystemExit: 2\n",
      "\n",
      "During handling of the above exception, another exception occurred:\n",
      "\n",
      "Traceback (most recent call last):\n",
      "  File \"/opt/anaconda3/lib/python3.9/site-packages/IPython/core/ultratb.py\", line 1101, in get_records\n",
      "    return _fixed_getinnerframes(etb, number_of_lines_of_context, tb_offset)\n",
      "  File \"/opt/anaconda3/lib/python3.9/site-packages/IPython/core/ultratb.py\", line 248, in wrapped\n",
      "    return f(*args, **kwargs)\n",
      "  File \"/opt/anaconda3/lib/python3.9/site-packages/IPython/core/ultratb.py\", line 281, in _fixed_getinnerframes\n",
      "    records = fix_frame_records_filenames(inspect.getinnerframes(etb, context))\n",
      "  File \"/opt/anaconda3/lib/python3.9/inspect.py\", line 1543, in getinnerframes\n",
      "    frameinfo = (tb.tb_frame,) + getframeinfo(tb, context)\n",
      "AttributeError: 'tuple' object has no attribute 'tb_frame'\n"
     ]
    },
    {
     "ename": "TypeError",
     "evalue": "object of type 'NoneType' has no len()",
     "output_type": "error",
     "traceback": [
      "\u001b[0;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[0;31mArgumentError\u001b[0m                             Traceback (most recent call last)",
      "\u001b[0;32m/opt/anaconda3/lib/python3.9/argparse.py\u001b[0m in \u001b[0;36mparse_known_args\u001b[0;34m(self, args, namespace)\u001b[0m\n\u001b[1;32m   1857\u001b[0m             \u001b[0;32mtry\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m-> 1858\u001b[0;31m                 \u001b[0mnamespace\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0margs\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0m_parse_known_args\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0margs\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mnamespace\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m   1859\u001b[0m             \u001b[0;32mexcept\u001b[0m \u001b[0mArgumentError\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
      "\u001b[0;32m/opt/anaconda3/lib/python3.9/argparse.py\u001b[0m in \u001b[0;36m_parse_known_args\u001b[0;34m(self, arg_strings, namespace)\u001b[0m\n\u001b[1;32m   2066\u001b[0m             \u001b[0;31m# consume the next optional and any arguments for it\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m-> 2067\u001b[0;31m             \u001b[0mstart_index\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mconsume_optional\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mstart_index\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m   2068\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n",
      "\u001b[0;32m/opt/anaconda3/lib/python3.9/argparse.py\u001b[0m in \u001b[0;36mconsume_optional\u001b[0;34m(start_index)\u001b[0m\n\u001b[1;32m   1988\u001b[0m                         \u001b[0mmsg\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0m_\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m'ignored explicit argument %r'\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m-> 1989\u001b[0;31m                         \u001b[0;32mraise\u001b[0m \u001b[0mArgumentError\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0maction\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mmsg\u001b[0m \u001b[0;34m%\u001b[0m \u001b[0mexplicit_arg\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m   1990\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n",
      "\u001b[0;31mArgumentError\u001b[0m: argument --shell-completion/-sc: ignored explicit argument '9002'",
      "\nDuring handling of the above exception, another exception occurred:\n",
      "\u001b[0;31mSystemExit\u001b[0m                                Traceback (most recent call last)",
      "    \u001b[0;31m[... skipping hidden 1 frame]\u001b[0m\n",
      "\u001b[0;32m/var/folders/k9/7_28zhp52b3f5wvy8fcyp8440000gn/T/ipykernel_9006/1060207763.py\u001b[0m in \u001b[0;36m<module>\u001b[0;34m\u001b[0m\n\u001b[1;32m      8\u001b[0m \u001b[0;32mif\u001b[0m \u001b[0m__name__\u001b[0m \u001b[0;34m==\u001b[0m \u001b[0;34m\"__main__\"\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m----> 9\u001b[0;31m     \u001b[0mmain\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m",
      "\u001b[0;32m/opt/anaconda3/lib/python3.9/site-packages/hydra/main.py\u001b[0m in \u001b[0;36mdecorated_main\u001b[0;34m(cfg_passthrough)\u001b[0m\n\u001b[1;32m     85\u001b[0m                 \u001b[0margs_parser\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mget_args_parser\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m---> 86\u001b[0;31m                 \u001b[0margs\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0margs_parser\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mparse_args\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m     87\u001b[0m                 \u001b[0;32mif\u001b[0m \u001b[0margs\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mexperimental_rerun\u001b[0m \u001b[0;32mis\u001b[0m \u001b[0;32mnot\u001b[0m \u001b[0;32mNone\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
      "\u001b[0;32m/opt/anaconda3/lib/python3.9/argparse.py\u001b[0m in \u001b[0;36mparse_args\u001b[0;34m(self, args, namespace)\u001b[0m\n\u001b[1;32m   1824\u001b[0m     \u001b[0;32mdef\u001b[0m \u001b[0mparse_args\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mself\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0margs\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0;32mNone\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mnamespace\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0;32mNone\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m-> 1825\u001b[0;31m         \u001b[0margs\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0margv\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mparse_known_args\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0margs\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mnamespace\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m   1826\u001b[0m         \u001b[0;32mif\u001b[0m \u001b[0margv\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
      "\u001b[0;32m/opt/anaconda3/lib/python3.9/argparse.py\u001b[0m in \u001b[0;36mparse_known_args\u001b[0;34m(self, args, namespace)\u001b[0m\n\u001b[1;32m   1860\u001b[0m                 \u001b[0merr\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0m_sys\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mexc_info\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m[\u001b[0m\u001b[0;36m1\u001b[0m\u001b[0;34m]\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m-> 1861\u001b[0;31m                 \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0merror\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mstr\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0merr\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m   1862\u001b[0m         \u001b[0;32melse\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
      "\u001b[0;32m/opt/anaconda3/lib/python3.9/argparse.py\u001b[0m in \u001b[0;36merror\u001b[0;34m(self, message)\u001b[0m\n\u001b[1;32m   2581\u001b[0m         \u001b[0margs\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0;34m{\u001b[0m\u001b[0;34m'prog'\u001b[0m\u001b[0;34m:\u001b[0m \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mprog\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0;34m'message'\u001b[0m\u001b[0;34m:\u001b[0m \u001b[0mmessage\u001b[0m\u001b[0;34m}\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m-> 2582\u001b[0;31m         \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mexit\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;36m2\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0m_\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m'%(prog)s: error: %(message)s\\n'\u001b[0m\u001b[0;34m)\u001b[0m \u001b[0;34m%\u001b[0m \u001b[0margs\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m",
      "\u001b[0;32m/opt/anaconda3/lib/python3.9/argparse.py\u001b[0m in \u001b[0;36mexit\u001b[0;34m(self, status, message)\u001b[0m\n\u001b[1;32m   2568\u001b[0m             \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0m_print_message\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mmessage\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0m_sys\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mstderr\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m-> 2569\u001b[0;31m         \u001b[0m_sys\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mexit\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mstatus\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m   2570\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n",
      "\u001b[0;31mSystemExit\u001b[0m: 2",
      "\nDuring handling of the above exception, another exception occurred:\n",
      "\u001b[0;31mTypeError\u001b[0m                                 Traceback (most recent call last)",
      "    \u001b[0;31m[... skipping hidden 1 frame]\u001b[0m\n",
      "\u001b[0;32m/opt/anaconda3/lib/python3.9/site-packages/IPython/core/interactiveshell.py\u001b[0m in \u001b[0;36mshowtraceback\u001b[0;34m(self, exc_tuple, filename, tb_offset, exception_only, running_compiled_code)\u001b[0m\n\u001b[1;32m   2068\u001b[0m                     stb = ['An exception has occurred, use %tb to see '\n\u001b[1;32m   2069\u001b[0m                            'the full traceback.\\n']\n\u001b[0;32m-> 2070\u001b[0;31m                     stb.extend(self.InteractiveTB.get_exception_only(etype,\n\u001b[0m\u001b[1;32m   2071\u001b[0m                                                                      value))\n\u001b[1;32m   2072\u001b[0m                 \u001b[0;32melse\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
      "\u001b[0;32m/opt/anaconda3/lib/python3.9/site-packages/IPython/core/ultratb.py\u001b[0m in \u001b[0;36mget_exception_only\u001b[0;34m(self, etype, value)\u001b[0m\n\u001b[1;32m    752\u001b[0m         \u001b[0mvalue\u001b[0m \u001b[0;34m:\u001b[0m \u001b[0mexception\u001b[0m \u001b[0mvalue\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    753\u001b[0m         \"\"\"\n\u001b[0;32m--> 754\u001b[0;31m         \u001b[0;32mreturn\u001b[0m \u001b[0mListTB\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mstructured_traceback\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mself\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0metype\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mvalue\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m    755\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    756\u001b[0m     \u001b[0;32mdef\u001b[0m \u001b[0mshow_exception_only\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mself\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0metype\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mevalue\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
      "\u001b[0;32m/opt/anaconda3/lib/python3.9/site-packages/IPython/core/ultratb.py\u001b[0m in \u001b[0;36mstructured_traceback\u001b[0;34m(self, etype, evalue, etb, tb_offset, context)\u001b[0m\n\u001b[1;32m    627\u001b[0m             \u001b[0mchained_exceptions_tb_offset\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0;36m0\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    628\u001b[0m             out_list = (\n\u001b[0;32m--> 629\u001b[0;31m                 self.structured_traceback(\n\u001b[0m\u001b[1;32m    630\u001b[0m                     \u001b[0metype\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mevalue\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0;34m(\u001b[0m\u001b[0metb\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mchained_exc_ids\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m,\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    631\u001b[0m                     chained_exceptions_tb_offset, context)\n",
      "\u001b[0;32m/opt/anaconda3/lib/python3.9/site-packages/IPython/core/ultratb.py\u001b[0m in \u001b[0;36mstructured_traceback\u001b[0;34m(self, etype, value, tb, tb_offset, number_of_lines_of_context)\u001b[0m\n\u001b[1;32m   1365\u001b[0m         \u001b[0;32melse\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m   1366\u001b[0m             \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mtb\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mtb\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m-> 1367\u001b[0;31m         return FormattedTB.structured_traceback(\n\u001b[0m\u001b[1;32m   1368\u001b[0m             self, etype, value, tb, tb_offset, number_of_lines_of_context)\n\u001b[1;32m   1369\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n",
      "\u001b[0;32m/opt/anaconda3/lib/python3.9/site-packages/IPython/core/ultratb.py\u001b[0m in \u001b[0;36mstructured_traceback\u001b[0;34m(self, etype, value, tb, tb_offset, number_of_lines_of_context)\u001b[0m\n\u001b[1;32m   1265\u001b[0m         \u001b[0;32mif\u001b[0m \u001b[0mmode\u001b[0m \u001b[0;32min\u001b[0m \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mverbose_modes\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m   1266\u001b[0m             \u001b[0;31m# Verbose modes need a full traceback\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m-> 1267\u001b[0;31m             return VerboseTB.structured_traceback(\n\u001b[0m\u001b[1;32m   1268\u001b[0m                 \u001b[0mself\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0metype\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mvalue\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mtb\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mtb_offset\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mnumber_of_lines_of_context\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m   1269\u001b[0m             )\n",
      "\u001b[0;32m/opt/anaconda3/lib/python3.9/site-packages/IPython/core/ultratb.py\u001b[0m in \u001b[0;36mstructured_traceback\u001b[0;34m(self, etype, evalue, etb, tb_offset, number_of_lines_of_context)\u001b[0m\n\u001b[1;32m   1122\u001b[0m         \u001b[0;34m\"\"\"Return a nice text document describing the traceback.\"\"\"\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m   1123\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m-> 1124\u001b[0;31m         formatted_exception = self.format_exception_as_a_whole(etype, evalue, etb, number_of_lines_of_context,\n\u001b[0m\u001b[1;32m   1125\u001b[0m                                                                tb_offset)\n\u001b[1;32m   1126\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n",
      "\u001b[0;32m/opt/anaconda3/lib/python3.9/site-packages/IPython/core/ultratb.py\u001b[0m in \u001b[0;36mformat_exception_as_a_whole\u001b[0;34m(self, etype, evalue, etb, number_of_lines_of_context, tb_offset)\u001b[0m\n\u001b[1;32m   1080\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m   1081\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m-> 1082\u001b[0;31m         \u001b[0mlast_unique\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mrecursion_repeat\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mfind_recursion\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0morig_etype\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mevalue\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mrecords\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m   1083\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m   1084\u001b[0m         \u001b[0mframes\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mformat_records\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mrecords\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mlast_unique\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mrecursion_repeat\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
      "\u001b[0;32m/opt/anaconda3/lib/python3.9/site-packages/IPython/core/ultratb.py\u001b[0m in \u001b[0;36mfind_recursion\u001b[0;34m(etype, value, records)\u001b[0m\n\u001b[1;32m    380\u001b[0m     \u001b[0;31m# first frame (from in to out) that looks different.\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    381\u001b[0m     \u001b[0;32mif\u001b[0m \u001b[0;32mnot\u001b[0m \u001b[0mis_recursion_error\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0metype\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mvalue\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mrecords\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m--> 382\u001b[0;31m         \u001b[0;32mreturn\u001b[0m \u001b[0mlen\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mrecords\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0;36m0\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m    383\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    384\u001b[0m     \u001b[0;31m# Select filename, lineno, func_name to track frames with\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
      "\u001b[0;31mTypeError\u001b[0m: object of type 'NoneType' has no len()"
     ]
    }
   ],
   "source": [
    "import hydra\n",
    "\n",
    "@hydra.main(config_path=\"conf\", config_name=\"config\")\n",
    "def main(conf):\n",
    "    for dataset in conf.dataset:\n",
    "        print(dataset)\n",
    "\n",
    "if __name__ == \"__main__\":\n",
    "    main()"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.9.13"
  },
  "orig_nbformat": 4
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
