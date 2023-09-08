def judge_prompts(coarse_person, prompt):
    caption_list = prompt.split()
    key = False
    index = -1
    for word in coarse_person:
        if key:
            break
        try:
            index = caption_list.index(word)
            key = True
        except:
            continue
    return key, index

def generate_augment_prompt(coarse_person, prompt, special_token, index=-1):
    caption_list = prompt.split()
    if index == -1:
        for word in coarse_person:
            if index != -1:
                break
            try:
                index = caption_list.index(word)
            except:
                continue
        assert index != -1, 'wrong prompt!\n'
    caption_list.insert(index+1, special_token)
    augment_prompt = ' '.join(caption_list)
    return augment_prompt

# test
# if __name__ == '__main__':
#     coarse_person = ['man', 'woman', 'guy', 'boy', 'girl', 'lady', 'sir', 'male', 'female', 'person', 'student']
#     prompt = 'a running man is near the beach'
#     special_token = 'sks'
#     print(generate_augment_prompt(coarse_person, prompt, special_token))
