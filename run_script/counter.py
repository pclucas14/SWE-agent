import argparse
from datasets import load_dataset


def filter_criterion(message):
    """
    Customizable filter function to determine if a message meets certain criteria.
    
    Args:
        message (dict): A message dictionary with 'role' and 'content' keys
        
    Returns:
        bool: True if the message meets the filter criteria, False otherwise
    """
    return message['role'] == 'assistant' and 'pytest' in message['content']


def count_messages(data_file, filter_func=filter_criterion):
    """
    Count messages in a dataset based on a filter function.
    
    Args:
        data_file (str): Path to the data file
        filter_func (callable): Function to filter messages
    """
    filtered_count = 0
    total_assistant_calls = 0
    
    # Load dataset
    dataset = load_dataset('json', data_files=data_file, split='train')

    for m in dataset['messages']:
        for j in m:
            if filter_func(j):
                filtered_count += 1
        total_assistant_calls += len(m)
    
    # Print results
    print(f"Total messages: {len(dataset['messages'])}")
    print(f"Total calls to assistant: {total_assistant_calls}")
    print(f"Total calls matching filter: {filtered_count}")
    print(f"Percentage of messages matching filter: {filtered_count / total_assistant_calls * 100:.2f}%")


def main():
    parser = argparse.ArgumentParser(description='Count messages in a dataset based on filter criteria')
    parser.add_argument(
        '--data-file', 
        type=str, 
        default='/home/zhengyanshi/project/SWE-agent/data/automated_pipeline_o3_bugs30_combos50_depth2_workers32_nbugs1_patches2_perfile2_permodule10/pylint-dev__pylint.1f8c4d9e.json',
        help='Path to the JSON data file'
    )
    
    args = parser.parse_args()
    count_messages(args.data_file)


if __name__ == "__main__":
    main()