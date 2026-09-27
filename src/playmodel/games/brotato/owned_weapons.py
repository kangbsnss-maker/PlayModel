"""Observed owned-weapon operations, separate from purchase and loot recycling."""
from itertools import permutations
import hashlib
import json
from pathlib import Path
import re
import time
import unicodedata

from .ocr import rows_in_region
from .menu import selected_button, navigation_key, BUTTONS
from .setup_run import focused_tile

SLOTS={i:(1131+106*i,845,1227+106*i,941) for i in range(3)}


def compact(text):
    return re.sub(r'\s+','',unicodedata.normalize('NFKC',text)).casefold()


def snapshot(shot,ocr,pixels):
    """First-row layout only; unknown layouts are never actionable."""
    header=''.join(rows_in_region(ocr,(0,0,400,120)))
    wave=re.search(r'wave(\d+)',compact(header))
    counts=re.search(r'weapons\((\d+)/(\d+)\)',compact(''.join(rows_in_region(ocr,(1100,760,1460,830)))))
    currency=ocr.get('_local_shop_currency')
    money=currency.get('currency') if isinstance(currency,dict) else getattr(currency,'currency',None)
    if money is None:
        raw=compact(''.join(rows_in_region(ocr,(815,25,1010,110))))
        money=int(raw) if re.fullmatch(r'\d{1,5}',raw) else None
    if not wave or not counts or not 1<=int(counts[1])<=3 or money is None:
        return None
    icons=[]
    icon_masks=[]
    for index in range(int(counts[1])):
        left,top,_,_=SLOTS[index]
        # Black outline mask removes hover background while retaining icon shape.
        mask=bytes(int(max(pixels[(y*1920+x)*4:(y*1920+x)*4+3])<10)
                   for y in range(top+12,top+84) for x in range(left+12,left+84))
        icons.append(hashlib.sha256(mask).hexdigest())
        icon_masks.append(hex(int(''.join(map(str,mask)),2)))
    return {'wave':int(wave[1]),'count':int(counts[1]),'capacity':int(counts[2]),
            'currency':money,'icons':icons,'icon_masks':icon_masks,'frame_ref':str(Path(shot['session_directory'])/'frame.png'),
            'frame_sha256':shot['frame_sha256'],'observed_at_ns':shot['capture_started_at_ns'],
            'available_at_ns':ocr['available_at_ns'],
            'target_identity':{k:shot.get(k) for k in ('hwnd','pid','executable')}}


def popup(ocr,pixels):
    words=[w for line in ocr['lines'] for w in line['words']]
    labels={}
    for word in words:
        text=compact(word['text'])
        action=('recycle' if text in ('recycle','recycie') else
                'cancel' if text=='cancel' else 'combine' if text=='combine' else None)
        if action and 700<word['x']<1450 and 400<word['y']<830:
            labels[action]=word
    if not {'recycle','cancel'}<=set(labels): return None
    cancel=labels['cancel']
    center=cancel['x']+cancel['width']/2
    boxes={key:(int(center-160),int(w['y']-5),int(center+160),int(w['y']+w['height']+7))
           for key,w in labels.items()}
    price_text=compact(''.join(rows_in_region(ocr,boxes['recycle'])))
    price=re.fullmatch(r'(?:recycle|recycie)\(\+(\d{1,5})\)',price_text)
    # Locate the name above the weapon-stat block, inside the same popup column.
    body=rows_in_region(ocr,(int(center-165),330,int(center+170),boxes['recycle'][1]))
    damage=next((i for i,t in enumerate(body) if compact(t).startswith('damage')),None)
    if damage is None or damage<2: return None
    title=body[damage-2]
    if not title or len(title)>70: return None
    sets=rows_in_region(ocr,(int(center+180),330,min(1900,int(center+535)),660))
    selection=selected_button(pixels,1920,1080,scene='shop',candidates=boxes)
    return {'name':title,'category':body[damage-1], 'body':body,'set_effects_raw':sets,'payout':int(price[1]) if price else None,
            'boxes':boxes,'selected':selection.selected_id,'tier_verified':False,
            'semantic_key':(compact(title),int(price[1]) if price else None,tuple(sorted(boxes)))}


def same(first,second):
    return bool(first and second and first['target_identity']==second['target_identity']
                and first['available_at_ns']<second['observed_at_ns']
                and second['observed_at_ns']-first['observed_at_ns']<=2_000_000_000
                and all(first[k]==second[k] for k in ('wave','count','currency','icons')))


def inventory_equal(before,after,indices):
    if len(indices)!=after['count']:return False
    def matches(i,j):
        if before['icons'][i]==after['icons'][j]:return True
        if not before.get('icon_masks') or not after.get('icon_masks'):return False
        a,b=int(before['icon_masks'][i],16),int(after['icon_masks'][j],16)
        # Live slot compaction changed one antialias pixel of 5184. Bound
        # tolerance to two bits; never compare only sparse/background masks.
        return min(a.bit_count(),b.bit_count())>=100 and (a^b).bit_count()<=2
    return any(all(matches(i,j) for j,i in enumerate(order)) for order in permutations(indices))


def verify_operation(before,first,second,*,action,slot,payout,sent_at_ns):
    if not same(first,second) or first['observed_at_ns']<=sent_at_ns: return False
    if second['wave']!=before['wave'] or second['target_identity']!=before['target_identity']: return False
    if action=='cancel': return inventory_equal(before,second,range(before['count'])) and second['currency']==before['currency']
    if before['count']<2 or not 0<=slot<before['count']: return False
    icon=before['icons'][slot]
    remaining=[i for i in range(before['count']) if i!=slot]
    if action=='recycle':
        return (inventory_equal(before,second,remaining) and second['count']==before['count']-1 and type(payout) is int
                and second['currency']==before['currency']+payout)
    if action=='combine':
        # Selected Combine + two identical silhouettes; tier still unverified.
        return (before['icons'].count(icon)>=2 and inventory_equal(before,second,remaining)
                and second['count']==before['count']-1 and second['currency']==before['currency'])
    return False


def validate_inventory_sources(observation, application):
    """Freeze source-backed inventory context again at the learning boundary."""
    sources=[]
    context=observation.get('owned_weapon_knowledge',{}).get('source')
    if context:
        sources.extend(context.get('owned_weapon_before_pair',[]))
        sources.append(context)
    if application.get('owned_operation'):
        pair=application.get('inventory_before_pair',[])
        auth=application.get('authorization_frame',{})
        if len(pair)!=2 or not same(*pair):
            raise ValueError('Owned operation requires independent inventory pair')
        if (not pair[-1]['available_at_ns']<=observation['observed_at_ns']
                or not observation['observed_at_ns']<=auth.get('observed_at_ns',0)
                <=auth.get('available_at_ns',0)<=application['sent_at_ns']
                or application['sent_at_ns']-auth['observed_at_ns']>750_000_000):
            raise ValueError('Owned authorization chronology invalid')
        sources.extend([*pair,auth])
        receipt=application.get('transport_receipt',{})
        path=Path(receipt.get('path',''))
        if hashlib.sha256(path.read_bytes()).hexdigest()!=receipt.get('sha256'):
            raise ValueError('Owned transport receipt changed')
        transport=json.loads(path.read_text(encoding='utf8'))
        if (transport.get('decision_id')!=application['decision_id']
                or transport.get('sent_at_ns')!=application['sent_at_ns']
                or transport.get('source')!=auth
                or transport.get('reason')!='owned_'+application['owned_operation']
                or transport.get('key')!='enter'):
            raise ValueError('Owned transport receipt mismatch')
    proofs=[]
    for source in sources:
        if (source['observed_at_ns']>application['sent_at_ns']
                or hashlib.sha256(Path(source['frame_ref']).read_bytes()).hexdigest()!=source['frame_sha256']):
            raise ValueError('Owned inventory source changed')
        proofs.append({'path':source['frame_ref'],'sha256':source['frame_sha256']})
    if application.get('owned_operation'):proofs.append(receipt)
    return proofs


class OwnedWeaponLearning:
    """One bounded inspection/decision per wave; caller is the sole input writer."""
    def __init__(self,menu):
        self.menu=menu
        self.done=set()
        self.previous=None
        self.before=None
        self.card=None
        self.decision=None
        self.sent_at=None
        self.after=None
        self.slot=0
        self.steps=0
        self.recycled=0
        self.memory={}
        self.last_key=None
        self.force_cancel=False
        self.before_pair=[]
        self.last_sent_at=0
        self.opened_card_key=None

    def interrupt(self, *, abandoned=False):
        if self.decision is not None and not abandoned:
            self.menu.client.discard(self.decision['decision_id'],'owned_observation_boundary')
        self.before=self.card=self.decision=self.sent_at=self.after=self.previous=None
        self.opened_card_key=None
        self.force_cancel=False
        self.memory={}
        self.steps=0

    def step(self,shot,ocr,pixels):
        now=time.perf_counter_ns()
        if now-shot['capture_started_at_ns']>750_000_000:return 'wait'
        if ocr.get('recognition_path')=='exact_image_cache' or 'cache_source' in ocr:return 'wait'
        if now-self.last_sent_at<180_000_000:return 'wait'
        if self.menu.pending_decision is not None:return None
        self.steps+=1
        seen=snapshot(shot,ocr,pixels)
        card=popup(ocr,pixels)
        if self.steps>120:
            if self.before:self.done.add(self.before['wave'])
            self.interrupt()
            return 'wait'
        if card and card['selected'] is not None and self.before is None:
            # An already open modal has no owned-slot authorization. Recover by
            # Cancel only; never turn this recovery input into a policy label.
            source={'observed_at_ns':shot['capture_started_at_ns'],
                    'available_at_ns':ocr['available_at_ns'],
                    'frame_ref':str(Path(shot['session_directory'])/'frame.png'),
                    'frame_sha256':shot['frame_sha256']}
            key=('enter' if card['selected']=='cancel' else
                 navigation_key(card['boxes'][card['selected']],card['boxes']['cancel']))
            return self.propose(key,'recover_unowned_modal',source)
        if card and self.before is not None:
            # The modal covers count/currency. Retain their pre-open evidence;
            # these are not new observations and cannot verify a completed sale.
            seen={**self.before,'frame_ref':str(Path(shot['session_directory'])/'frame.png'),
                  'frame_sha256':shot['frame_sha256'],'observed_at_ns':shot['capture_started_at_ns'],
                  'available_at_ns':ocr['available_at_ns'], 'masked_inventory_fields':True,
                  'target_identity':{k:shot.get(k) for k in ('hwnd','pid','executable')}}
            if seen['target_identity']!=self.before['target_identity']:
                raise ValueError('Owned weapon target changed')
        if self.sent_at is not None and card and card['selected'] is None:
            if seen is None:
                seen={**self.before,'observed_at_ns':shot['capture_started_at_ns'],
                      'available_at_ns':ocr['available_at_ns'],
                      'frame_ref':str(Path(shot['session_directory'])/'frame.png'),'frame_sha256':shot['frame_sha256']}
            return self.propose('right','close_owned_hover',seen)
        if seen is None:
            if self.before is not None and self.steps>80:
                self.done.add(self.before['wave'])
                self.interrupt()
                return None
            return None if self.before is None else 'wait'
        if self.sent_at is not None:
            if card and card['selected'] is not None:
                if self.steps>80 and not self.force_cancel:
                    self.menu.client.discard(self.decision['decision_id'],'owned_modal_result_unverified')
                    self.decision=self.sent_at=None
                    self.force_cancel=True
                else:
                    return 'wait'
        if self.sent_at is not None:
            if verify_operation(self.before,self.after,seen,action=self.decision['action_id'],slot=self.slot,
                                payout=self.card['payout'],sent_at_ns=self.sent_at):
                self.commit(seen)
                return 'wait'
            if self.steps>80 and same(self.after,seen):
                # A stable normal shop is safe to re-enter even when this
                # operation's precise result cannot be credited to learning.
                self.menu.client.discard(self.decision['decision_id'],'owned_result_unverified_stable_shop')
                self.done.add(seen['wave'])
                self.interrupt(abandoned=True)
                return None
            self.after=seen
            return 'wait'
        if self.before is None:
            if seen['wave'] in self.done:return None
            # Inspection may cancel an existing owned popup, but never sell the last weapon.
            if not same(self.previous,seen):
                self.previous=seen
                return 'wait'
            self.before=seen
            self.before_pair=[self.previous,seen]
            self.steps=0
        if seen['wave']!=self.before['wave']:
            self.interrupt()
            return 'wait'
        if self.steps>80 and not self.force_cancel:
            if self.decision:
                self.menu.client.discard(self.decision['decision_id'],'owned_navigation_budget')
            self.decision=None
            self.force_cancel=True
            if card is None or card['selected'] is None:
                self.done.add(seen['wave'])
                self.interrupt()
                return None
        focused=focused_tile(pixels,1920,{i:SLOTS[i] for i in range(seen['count'])})
        if card is None or card['selected'] is None:
            if focused==self.slot and card:
                if self.card and self.card['semantic_key']==card['semantic_key']:
                    return self.propose('enter','open_owned_weapon',seen)
                self.card=card
                return 'wait'
            boxes=dict(BUTTONS['shop'])
            if any('go' in text.casefold() for text in rows_in_region(ocr,(1450,950,1910,1070))):
                boxes['depart']=(1484,981,1884,1048)
            boxes.update({f'owned_{i}':SLOTS[i] for i in range(seen['count'])})
            focus=selected_button(pixels,1920,1080,scene='shop',candidates=boxes)
            if focus.rect is None:return 'wait'
            return self.propose(navigation_key(focus.rect,SLOTS[self.slot],scene='shop'),'inspect_owned_weapon',seen)
        if self.card is None or self.card['semantic_key']!=card['semantic_key']:
            if self.decision is not None:
                self.menu.client.discard(self.decision['decision_id'],'owned_card_changed_before_confirmation')
                self.decision=None
            self.card=card
            return 'wait'
        if self.decision is None:
            options={'cancel':'Keep this weapon; cancel the menu'}
            owns_card=self.opened_card_key==card['semantic_key']
            if owns_card and not self.force_cancel and self.before['count']>1 and card['payout'] is not None:
                options['recycle']=f'Recycle {card["name"]} for {card["payout"]} materials; lose this weapon'
            if owns_card and not self.force_cancel and 'combine' in card['boxes'] and self.before['icons'].count(self.before['icons'][self.slot])>=2:
                options['combine']='Combine matching owned weapons; verify inventory change; tier not yet calibrated'
            source={'frame_ref':seen['frame_ref'],'frame_sha256':seen['frame_sha256'],
                    'observed_at_ns':seen['observed_at_ns'],'available_at_ns':seen['available_at_ns'],
                    'run_id':self.menu.recorder.run_id,'game_build_id':self.menu.recorder.game_build_id,
                    'owned_weapon_before':self.before,'owned_weapon_card':card,
                    'owned_weapon_before_pair':self.before_pair,
                    'candidates':[{'candidate_id':k,'target':k,'legal':True} for k in options]}
            character=getattr(self.menu.recorder,'character_context',{})
            if character:source['character_context']=character
            state={'game':'Brotato','scene':'owned_weapon','wave':seen['wave'],
                   'character':character.get('name'),'weapon':card['name'],'currency':seen['currency'],
                   'weapon_count':seen['count'],'recycled_this_run':self.recycled,
                   'goal':'survive; recycle 12 weapons in one run if viable',
                   'set_observation':' '.join(card['set_effects_raw'])}
            from .character_context import affinity
            state['character_weapon_fit']=affinity(character.get('traits',[]),' '.join(card['body']))
            self.decision=self.menu.client.choose(state,options,source)
            self.card=card
            self.memory={'weapon':card['name'],'category':card['category'],
                         'source':source, 'set_effects_raw':card['set_effects_raw'],
                         'tier_verified':False,'recycled_this_run':self.recycled}
            return 'wait'
        action=self.decision['action_id']
        if action not in card['boxes']:return 'wait'
        key='enter' if card['selected']==action else navigation_key(card['boxes'][card['selected']],card['boxes'][action])
        return self.propose(key,'owned_'+action,seen)

    def propose(self,key,reason,seen):
        if key is None:return 'wait'
        self.last_key=(key,reason,seen)
        return key

    def sent(self,started,finished):
        key,reason,seen=self.last_key
        if not seen['observed_at_ns']<=started<=finished or started-seen['observed_at_ns']>750_000_000:
            raise ValueError('Owned weapon input was stale')
        self.last_sent_at=finished
        if self.menu.output_directory:
            path=self.menu.output_directory/'owned-inputs.jsonl'
            record={'key':key,'reason':reason,'source':seen,
                    'decision_id':self.decision['decision_id'] if self.decision else None,
                    'send_started_at_ns':started,'sent_at_ns':finished,'successful_transport_reported':True}
            with path.open('a',encoding='utf8') as stream:
                stream.write(json.dumps(record)+'\n')
            if key=='enter' and reason.startswith('owned_'):
                receipt_path=self.menu.output_directory/f'owned-transport-{finished}.json'
                with receipt_path.open('x',encoding='utf8') as stream:
                    json.dump(record,stream)
                self.receipt={'path':str(receipt_path),
                    'sha256':hashlib.sha256(receipt_path.read_bytes()).hexdigest()}
        if key=='enter' and reason=='open_owned_weapon':
            self.opened_card_key=self.card['semantic_key']
        elif key=='enter' and reason!='recover_unowned_modal':
            self.sent_at=finished
            self.authorization=dict(seen)

    def commit(self,seen):
        action=self.decision['action_id']
        decision_id=self.decision['decision_id']
        record_path=self.menu.client.output/f'choice-{decision_id}.json'
        record=json.loads(record_path.read_text(encoding='utf8'))
        for source in (*self.before_pair,self.after,seen):
            if hashlib.sha256(Path(source['frame_ref']).read_bytes()).hexdigest()!=source['frame_sha256']:
                raise ValueError('Owned weapon source changed')
        frames=[{'frame_ref':r['frame_ref'],'frame_sha256':r['frame_sha256']} for r in (self.after,seen)]
        proof={'decision_id':decision_id,'accepted':True,'successful_transport_reported':True,
               'game_application_verified':True,'actual_target':action,'sent_at_ns':self.sent_at,
               'verified_at_ns':time.perf_counter_ns(),'before_frame_ref':record['evidence']['frame_ref'],
               'after_frames':frames,'after_frame_ids':[r['frame_ref'] for r in frames],
               'authorization':{'sent_at_ns':self.sent_at,'target':action},
               'owned_operation':action,'inventory_before':self.before,'inventory_after':seen,
               'inventory_before_pair':self.before_pair,'authorization_frame':self.authorization,
               'transport_receipt':self.receipt,
               'tier_upgrade_verified':False,'bc_label':False}
        validate_inventory_sources(record['evidence'],proof)
        self.menu.client.accept(decision_id,proof)
        if self.menu.economy:
            record['_source_path']=str(record_path)
            self.menu.economy.record_purchase(record,proof)
        if self.menu.recorder.build_state:
            self.menu.recorder.build_state.invalidate_stats('owned_weapon_operation_reobserve')
        if action=='recycle':self.recycled+=1
        self.memory['recycled_this_run']=self.recycled
        self.memory['still_owned']=action=='cancel'
        self.done.add(seen['wave'])
        self.before=self.card=self.decision=self.sent_at=self.after=None
        self.previous=None
        self.force_cancel=False
        self.opened_card_key=None
